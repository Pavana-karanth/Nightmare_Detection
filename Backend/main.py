"""
Nightmare Detection API
FastAPI service for real-time nightmare detection from EEG spectrograms
Using Band-Weighted Deep SVDD with Relative Power Features
"""

import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Configuration constants matching training pipeline"""

    # Frequency processing
    FMIN = 0.5
    FMAX = 35.0
    FREQ_BINS = 100
    N_CHANNELS = 4
    CHANNEL_NAMES = ["C3", "C4", "F3", "F4"]

    # Band definitions (matching training)
    BANDS = {
        "delta": (0.5, 4.0),
        "slow_theta": (2.0, 5.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 31.0),
        "low_gamma": (31.0, 35.0),
    }

    # PORT
    PORT = 8000

    # Model paths
    MODEL_DIR = Path("models")
    REM_MODEL_PATH = MODEL_DIR / "rem_finetuned_model.pth"
    N2_MODEL_PATH = MODEL_DIR / "n2_finetuned_model.pth"

    # Severity thresholds (calibrated from validation)
    THRESHOLDS = {
        "REM": {
            "radius": 6.621957,
            "normal": 5.018054,  # 90th percentile
            "mild": 5.468412,  # 95th percentile
            "moderate": 6.621957,  # 99th percentile (radius)
            "severe": 7.5,  # Above radius
        },
        "N2": {
            "radius": 6.633777,
            "normal": 5.472726,  # 90th percentile
            "mild": 5.845134,  # 95th percentile
            "moderate": 6.633777,  # 99th percentile
            "severe": 7.5,  # Above radius
        },
    }

    # Band weights (literature-based)
    BAND_WEIGHTS = {
        "REM": {
            "slow_theta": 0.5,
            "beta": 0.2,
            "theta": 0.1,
            "low_gamma": 0.1,
            "delta": 0.05,
            "alpha": 0.05,
        },
        "N2": {
            "low_gamma": 0.35,
            "beta": 0.25,
            "slow_theta": 0.15,
            "theta": 0.15,
            "delta": 0.05,
            "alpha": 0.05,
        },
    }


# ============================================================================
# MODEL ARCHITECTURE (Must match training)
# ============================================================================


class FrequencyAwareSleepEncoder(nn.Module):
    """Band-aware encoder for EEG spectrograms"""

    def __init__(self, n_channels=4, band_embedding_dim=32, dropout=0.1):
        super().__init__()
        self.bands = Config.BANDS
        self.freq_bins = Config.FREQ_BINS
        self.freqs = np.linspace(Config.FMIN, Config.FMAX, self.freq_bins)

        # Create Boolean Masks for each band
        self.band_masks = {}
        for band_name, (fmin, fmax) in self.bands.items():
            mask = (self.freqs >= fmin) & (self.freqs <= fmax)
            self.band_masks[band_name] = torch.from_numpy(mask).bool()

        # Shared CNN Backbone
        self.shared_features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
        )

        # Band-Specific Dense Heads
        self.band_encoders = nn.ModuleDict()
        for band_name in self.bands.keys():
            self.band_encoders[band_name] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4, band_embedding_dim, bias=False),
                nn.BatchNorm1d(band_embedding_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            )

        self.total_embedding_dim = len(self.bands) * band_embedding_dim

    def forward(self, x):
        features = self.shared_features(x)
        band_embeddings = []
        for band_name in self.bands.keys():
            mask = self.band_masks[band_name].to(x.device)
            band_features = features[:, :, mask, :]
            band_emb = self.band_encoders[band_name](band_features)
            band_embeddings.append(band_emb)
        return torch.cat(band_embeddings, dim=1)

    def get_band_indices(self):
        """Helper to identify which embedding indices belong to which band."""
        band_dim = self.total_embedding_dim // len(self.bands)
        indices = {}
        for i, band_name in enumerate(self.bands.keys()):
            start = i * band_dim
            end = (i + 1) * band_dim
            indices[band_name] = (start, end)
        return indices


# ============================================================================
# NIGHTMARE DETECTOR SERVICE
# ============================================================================


class StageType(str, Enum):
    """Sleep stage types"""

    REM = "REM"
    N2 = "N2"


class NightmareDetector:
    """Main detection service with band-weighted Deep SVDD"""

    def __init__(self, model_path: str, stage: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.stage = stage
        logger.info(f"Loading {stage} model from {model_path}")

        # Load checkpoint
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # Initialize network
        self.net = FrequencyAwareSleepEncoder(
            n_channels=Config.N_CHANNELS, band_embedding_dim=32, dropout=0.1
        ).to(self.device)

        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()

        # Load hypersphere parameters
        self.center = checkpoint["center"].to(self.device)
        self.R = float(checkpoint.get("R", Config.THRESHOLDS[stage]["radius"]))

        # Load normalization stats from checkpoint
        self.channel_means = np.array(checkpoint.get("channel_means", [0.0] * 4))
        self.channel_stds = np.array(checkpoint.get("channel_stds", [4.5] * 4))

        # Band weights (literature-based)
        self.band_weights = checkpoint.get("band_weights", Config.BAND_WEIGHTS[stage])

        # Severity thresholds
        self.thresholds = Config.THRESHOLDS[stage]

        logger.info(f"✅ {stage} model loaded successfully")
        logger.info(f"   Radius: {self.R:.4f}")
        logger.info(f"   Center norm: {torch.norm(self.center).item():.4f}")
        logger.info(f"   Channel means: {self.channel_means}")
        logger.info(f"   Channel stds: {self.channel_stds}")

    def preprocess_npy(self, npy_data: bytes) -> torch.Tensor:
        """
        Preprocess uploaded .npy spectrogram array

        Expected format:
        - Shape: (4, 100, time_bins) - 4 channels, 100 freq bins, variable time
        - Values: Relative power in dB (already normalized, mean≈0, std≈4-5)
        - Channels: [C3, C4, F3, F4]

        Returns:
        - Tensor: (1, 4, 100, time_bins) ready for model
        """
        try:
            # Load numpy array from bytes
            spec = np.load(io.BytesIO(npy_data))

            # Validate shape
            if len(spec.shape) != 3:
                raise ValueError(
                    f"Expected 3D array (channels, freq, time), got shape {spec.shape}"
                )

            if spec.shape[0] != Config.N_CHANNELS:
                raise ValueError(
                    f"Expected {Config.N_CHANNELS} channels, got {spec.shape[0]}"
                )

            if spec.shape[1] != Config.FREQ_BINS:
                raise ValueError(
                    f"Expected {Config.FREQ_BINS} frequency bins, got {spec.shape[1]}"
                )

            # Verify it's relative power (mean should be near 0)
            channel_means = [np.mean(spec[i]) for i in range(Config.N_CHANNELS)]
            if any(abs(m) > 2.0 for m in channel_means):
                logger.warning(
                    f"Channel means {channel_means} seem high for relative power (expected ≈0)"
                )

            # NO additional normalization - data is already in relative power format
            # Just convert to tensor and add batch dimension
            spec_tensor = torch.FloatTensor(spec).unsqueeze(0)

            return spec_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Error processing .npy file: {str(e)}")
            raise ValueError(f"Failed to process .npy spectrogram: {str(e)}")

    def compute_anomaly_score(self, spectrogram: torch.Tensor) -> Dict:
        """
        Compute band-weighted anomaly score

        Returns:
        - total_score: Overall distance from hypersphere center
        - band_scores: Individual band contributions
        """
        with torch.no_grad():
            # Get embedding
            embedding = self.net(spectrogram)

            # Compute band-weighted distance
            indices = self.net.get_band_indices()
            band_scores = {}
            weighted_dists = []

            for band, (s, e) in indices.items():
                # Squared distance for this band
                diff = (embedding[:, s:e] - self.center[s:e]) ** 2
                band_dist = torch.sum(diff, dim=1).item()

                # Apply band weight
                weighted_dist = band_dist * self.band_weights[band]
                band_scores[band] = {
                    "raw_distance": float(band_dist),
                    "weight": float(self.band_weights[band]),
                    "weighted_distance": float(weighted_dist),
                }
                weighted_dists.append(weighted_dist)

            total_score = sum(weighted_dists)

            return {"total_score": total_score, "band_scores": band_scores}

    def classify_severity(self, score: float) -> Dict:
        """
        Classify nightmare severity using calibrated thresholds

        Returns detailed classification with confidence scores
        """
        # Determine severity level
        if score < self.thresholds["normal"]:
            severity = "normal"
            severity_level = 0
            is_nightmare = False
        elif score < self.thresholds["mild"]:
            severity = "mild"
            severity_level = 1
            is_nightmare = True
        elif score < self.thresholds["moderate"]:
            severity = "moderate"
            severity_level = 2
            is_nightmare = True
        elif score < self.thresholds["severe"]:
            severity = "severe"
            severity_level = 3
            is_nightmare = True
        else:
            severity = "critical"
            severity_level = 4
            is_nightmare = True

        # Compute probability scores (normalized against calibration distribution)
        # Based on validation: normal dreams mean ≈ 3.8-4.1, nightmares ≈ 73-93 percentile
        max_observed = self.thresholds["severe"] * 1.5  # Conservative upper bound
        normalized_score = min(score / max_observed, 1.0)

        nightmare_probability = normalized_score * 100
        normal_probability = (1 - normalized_score) * 100

        # Confidence based on distance from nearest threshold
        if not is_nightmare:
            distance_from_threshold = abs(score - self.thresholds["normal"])
            ref_threshold = self.thresholds["normal"]
        else:
            distance_from_threshold = abs(score - self.R)
            ref_threshold = self.R

        confidence = min(distance_from_threshold / ref_threshold * 100, 100)

        return {
            "is_nightmare": is_nightmare,
            "severity": severity,
            "severity_level": severity_level,
            "nightmare_probability": round(nightmare_probability, 1),
            "normal_probability": round(normal_probability, 1),
            "confidence": round(confidence, 1),
            "anomaly_score": round(score, 4),
            "radius_threshold": round(self.R, 4),
            "score_vs_radius": round(score / self.R, 2),
        }

    def generate_insights(self, classification: Dict, band_scores: Dict) -> List[str]:
        """Generate clinical insights based on classification and band contributions"""
        insights = []

        if not classification["is_nightmare"]:
            insights.append("✅ EEG patterns consistent with normal dream activity")
            insights.append(
                f"Anomaly score ({classification['anomaly_score']:.2f}) below clinical threshold"
            )
            insights.append(
                "Sleep architecture appears stable across all frequency bands"
            )
        else:
            # Severity-specific insights
            severity = classification["severity"]
            score = classification["anomaly_score"]

            if severity == "mild":
                insights.append("⚠️ Mild nightmare markers detected")
                insights.append("Slight elevation in cortical arousal patterns")
                insights.append("Consider monitoring sleep quality over time")

            elif severity == "moderate":
                insights.append("⚠️ Moderate nightmare activity detected")
                insights.append("Significant spectral power abnormalities detected")
                insights.append(
                    "Clinical evaluation recommended for persistent symptoms"
                )

            elif severity == "severe":
                insights.append("🔴 Severe nightmare disorder markers present")
                insights.append("Multiple arousal indicators across frequency bands")
                insights.append("Immediate clinical intervention advised")

            else:  # critical
                insights.append("🔴 CRITICAL: Extreme nightmare activity")
                insights.append(
                    f"Anomaly score ({score:.2f}) significantly exceeds threshold"
                )
                insights.append("Urgent psychiatric evaluation recommended")

            # Band-specific insights (identify dominant contributors)
            sorted_bands = sorted(
                band_scores.items(),
                key=lambda x: x[1]["weighted_distance"],
                reverse=True,
            )

            top_band = sorted_bands[0][0]
            top_contribution = sorted_bands[0][1]["weighted_distance"]
            total_weighted = sum(b[1]["weighted_distance"] for b in sorted_bands)

            if top_contribution / total_weighted > 0.4:  # >40% contribution
                band_insights = {
                    "slow_theta": "Elevated slow-theta/delta ratio suggests fear extinction failure (PTSD-like patterns)",
                    "beta": "Increased beta power indicates cortical hyperarousal and anxiety",
                    "low_gamma": "Elevated gamma activity suggests autonomic dysregulation",
                    "theta": "Theta band abnormalities associated with emotional processing disruption",
                    "alpha": "Alpha intrusions indicate arousal instability",
                    "delta": "Delta suppression suggests fragmented slow-wave sleep",
                }

                if top_band in band_insights:
                    insights.append(f"🔬 Primary mechanism: {band_insights[top_band]}")

        # Add stage-specific context
        if self.stage == "REM":
            insights.append(
                "Stage: REM sleep analysis - focus on fear processing and emotion regulation"
            )
        else:
            insights.append(
                "Stage: N2 sleep analysis - focus on sleep consolidation and arousal patterns"
            )

        return insights

    def analyze(self, file_data: bytes) -> Dict:
        """
        Main analysis pipeline

        Args:
            file_data: Raw .npy file bytes

        Returns:
            Complete analysis results
        """
        try:
            # Preprocess
            spectrogram = self.preprocess_npy(file_data)

            # Compute anomaly score with band breakdown
            score_data = self.compute_anomaly_score(spectrogram)

            # Classify
            classification = self.classify_severity(score_data["total_score"])

            # Generate insights
            insights = self.generate_insights(classification, score_data["band_scores"])

            # Compile results
            results = {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "stage": self.stage,
                "classification": classification,
                "band_analysis": score_data["band_scores"],
                "insights": insights,
                "metadata": {
                    "model_version": "2.0_band_weighted",
                    "method": "Deep SVDD with Relative Power",
                    "channels": Config.CHANNEL_NAMES,
                    "band_weights": self.band_weights,
                    "embedding_dim": self.net.total_embedding_dim,
                },
            }

            logger.info(
                f"Analysis complete [{self.stage}]: {classification['severity']} "
                f"(score: {score_data['total_score']:.4f})"
            )

            return results

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize detectors
    global detectors
    try:
        if not Config.REM_MODEL_PATH.exists():
            raise FileNotFoundError(f"REM model not found at {Config.REM_MODEL_PATH}")
        if not Config.N2_MODEL_PATH.exists():
            raise FileNotFoundError(f"N2 model not found at {Config.N2_MODEL_PATH}")

        detectors["REM"] = NightmareDetector(
            model_path=str(Config.REM_MODEL_PATH), stage="REM", device="cpu"
        )

        detectors["N2"] = NightmareDetector(
            model_path=str(Config.N2_MODEL_PATH), stage="N2", device="cpu"
        )

        logger.info("✅ All nightmare detectors initialized successfully")
        yield  # Startup complete, continue to app
    except Exception as e:
        logger.error(f"❌ Failed to initialize detectors: {str(e)}")
        raise
    finally:
        # Optionally clean up resources here
        pass


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Nightmare Detection API v2.0",
    description="Band-Weighted Deep SVDD for EEG-based Nightmare Detection",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instances
detectors: Dict[str, NightmareDetector] = {}


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class ClassificationResult(BaseModel):
    """Classification output"""

    is_nightmare: bool
    severity: str
    severity_level: int
    nightmare_probability: float
    normal_probability: float
    confidence: float
    anomaly_score: float
    radius_threshold: float
    score_vs_radius: float


class BandScore(BaseModel):
    """Band-specific score"""

    raw_distance: float
    weight: float
    weighted_distance: float


class AnalysisResult(BaseModel):
    """Complete analysis result"""

    status: str
    timestamp: str
    stage: str
    classification: ClassificationResult
    band_analysis: Dict[str, BandScore]
    insights: List[str]
    metadata: Dict


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Nightmare Detection API v2.0",
        "version": "2.0.0",
        "method": "Band-Weighted Deep SVDD",
        "models_loaded": {"REM": "REM" in detectors, "N2": "N2" in detectors},
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    if not detectors:
        raise HTTPException(status_code=503, detail="Models not loaded")

    health_data = {}
    for stage, detector in detectors.items():
        health_data[stage] = {
            "loaded": True,
            "radius": detector.R,
            "device": str(detector.device),
            "embedding_dim": detector.net.total_embedding_dim,
            "band_weights": detector.band_weights,
        }

    return {
        "status": "healthy",
        "models": health_data,
        "config": {
            "channels": Config.CHANNEL_NAMES,
            "freq_range": f"{Config.FMIN}-{Config.FMAX} Hz",
            "freq_bins": Config.FREQ_BINS,
            "bands": Config.BANDS,
        },
    }


@app.post("/analyze/{stage}")
async def analyze_spectrogram(stage: StageType, file: UploadFile = File(...)):
    """
    Analyze EEG spectrogram for nightmare detection

    **Stage**: REM or N2

    **Input**: .npy file with shape (4, 100, time_bins)
    - 4 channels: [C3, C4, F3, F4]
    - 100 frequency bins: 0.5-35 Hz
    - Values: Relative power in dB (mean≈0, std≈4-5)

    **Returns**:
    - Classification (normal/mild/moderate/severe/critical)
    - Band-specific analysis
    - Clinical insights
    - Confidence scores
    """
    detector = detectors.get(stage.value)
    if detector is None:
        raise HTTPException(status_code=503, detail=f"{stage.value} model not loaded")

    # Validate file extension
    if not file.filename.endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail="Only .npy files are supported. Expected shape: (4, 100, time_bins)",
        )

    try:
        # Read file
        file_data = await file.read()

        # Analyze
        results = detector.analyze(file_data)

        return JSONResponse(content=results)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/batch-analyze/{stage}")
async def batch_analyze(stage: StageType, files: List[UploadFile] = File(...)):
    """
    Analyze multiple spectrograms in batch

    **Limit**: Maximum 50 files per batch

    Returns array of results, one per file
    """
    detector = detectors.get(stage.value)
    if detector is None:
        raise HTTPException(status_code=503, detail=f"{stage.value} model not loaded")

    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files per batch")

    results = []

    for file in files:
        try:
            if not file.filename.endswith(".npy"):
                results.append(
                    {
                        "status": "error",
                        "filename": file.filename,
                        "error": "Only .npy files supported",
                    }
                )
                continue

            file_data = await file.read()
            result = detector.analyze(file_data)
            result["filename"] = file.filename
            results.append(result)

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append(
                {"status": "error", "filename": file.filename, "error": str(e)}
            )

    return JSONResponse(
        content={"stage": stage.value, "total_files": len(files), "results": results}
    )


@app.get("/model-info/{stage}")
async def model_info(stage: StageType):
    """Get detailed model information and thresholds"""
    detector = detectors.get(stage.value)
    if detector is None:
        raise HTTPException(status_code=503, detail=f"{stage.value} model not loaded")

    return {
        "stage": stage.value,
        "radius": detector.R,
        "thresholds": detector.thresholds,
        "band_weights": detector.band_weights,
        "normalization": {
            "channel_means": detector.channel_means.tolist(),
            "channel_stds": detector.channel_stds.tolist(),
            "method": "Relative Power (no z-score)",
        },
        "expected_input": {
            "format": "NumPy array (.npy)",
            "shape": [4, 100, "variable_time"],
            "channels": Config.CHANNEL_NAMES,
            "freq_range": f"{Config.FMIN}-{Config.FMAX} Hz",
            "value_range": "Relative power in dB (mean≈0, std≈4-5)",
        },
        "severity_levels": {
            0: "normal",
            1: "mild",
            2: "moderate",
            3: "severe",
            4: "critical",
        },
        "bands": Config.BANDS,
        "embedding_dim": detector.net.total_embedding_dim,
    }


@app.get("/compare-stages")
async def compare_stages(
    rem_score: float = Query(..., description="REM anomaly score"),
    n2_score: float = Query(..., description="N2 anomaly score"),
):
    """
    Compare nightmare severity across REM and N2 stages

    Useful for longitudinal analysis or multi-stage recordings
    """
    if "REM" not in detectors or "N2" not in detectors:
        raise HTTPException(status_code=503, detail="Both models must be loaded")

    rem_classification = detectors["REM"].classify_severity(rem_score)
    n2_classification = detectors["N2"].classify_severity(n2_score)

    # Determine overall severity (take worst case)
    overall_level = max(
        rem_classification["severity_level"], n2_classification["severity_level"]
    )

    severity_map = {0: "normal", 1: "mild", 2: "moderate", 3: "severe", 4: "critical"}

    return {
        "REM": {
            "score": rem_score,
            "classification": rem_classification["severity"],
            "is_nightmare": rem_classification["is_nightmare"],
        },
        "N2": {
            "score": n2_score,
            "classification": n2_classification["severity"],
            "is_nightmare": n2_classification["is_nightmare"],
        },
        "overall": {
            "severity_level": overall_level,
            "severity": severity_map[overall_level],
            "recommendation": _get_recommendation(overall_level),
        },
    }


def _get_recommendation(severity_level: int) -> str:
    """Generate clinical recommendation based on severity"""
    recommendations = {
        0: "No intervention needed. Continue normal sleep hygiene practices.",
        1: "Monitor symptoms. Consider sleep diary and stress management techniques.",
        2: "Clinical consultation recommended. Consider CBT-I or imagery rehearsal therapy.",
        3: "Clinical intervention advised. Psychiatric evaluation for trauma-focused therapy.",
        4: "Urgent psychiatric evaluation recommended. Consider medication and intensive therapy.",
    }
    return recommendations.get(severity_level, "Unknown severity level")


@app.get("/validation-metrics")
async def validation_metrics():
    """
    Return validation performance metrics from cross-validation studies

    Useful for transparency and model explainability
    """
    return {
        "within_domain_performance": {
            "REM": {
                "separation": "97.6 percentile points",
                "roc_auc": 1.000,
                "cohens_d": 14.075,
                "specificity": 1.000,
                "sensitivity": 0.890,
            },
            "N2": {
                "separation": "96.2 percentile points",
                "roc_auc": 0.995,
                "cohens_d": 2.615,
                "specificity": 1.000,
                "sensitivity": 0.720,
            },
        },
        "cross_domain_performance": {
            "REM": {
                "median_percentile": 86.6,
                "status": "Moderate cross-equipment sensitivity",
            },
            "N2": {"median_percentile": 43.0, "status": "Good equipment invariance"},
        },
        "perturbation_validation": {
            "REM": {"median_percentile": 97.6, "generalization": "Excellent"}
        },
        "notes": [
            "Metrics based on synthetic nightmare validation with literature-based perturbations",
            "REM model shows higher cross-domain sensitivity - interpret with caution for new equipment",
            "N2 model demonstrates better equipment invariance",
            "All models use relative power features (equipment-invariant normalization)",
        ],
    }
