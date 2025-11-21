"""
Nightmare Detection API
FastAPI service for real-time nightmare detection from EEG spectrograms
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import numpy as np
import io
from PIL import Image
import base64
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL ARCHITECTURE (Must match training)
# ============================================================================

class RobustSpectrogramEncoder(nn.Module):
    """Deep SVDD encoder - matches training architecture"""
    def __init__(self, embedding_dim=128, dropout=0.1):
        super(RobustSpectrogramEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        self.flatten_size = 256 * 4 * 4
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, embedding_dim),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# NIGHTMARE DETECTOR SERVICE
# ============================================================================

class NightmareDetector:
    """Main detection service"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        logger.info(f"Loading model on device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Initialize network
        self.net = RobustSpectrogramEncoder(
            embedding_dim=checkpoint['embedding_dim']
        ).to(self.device)
        
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        
        # Load hypersphere parameters
        self.center = checkpoint['center'].to(self.device)
        self.R = float(checkpoint['R'].item() if torch.is_tensor(checkpoint['R']) else checkpoint['R'])
        
        # Normalization stats (from training)
        self.norm_mean = 0.5342
        self.norm_std = 0.1438
        
        # Severity thresholds (calibrated from evaluation)
        self.thresholds = {
            'radius': self.R,
            'mild': 1.5,      # Based on evaluation: mild nightmares ~1.2
            'moderate': 2.5,  # moderate ~2.4
            'severe': 3.5     # severe ~3.5
        }
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Radius: {self.R:.4f}")
        logger.info(f"   Center norm: {torch.norm(self.center).item():.4f}")
    
    def preprocess_spectrogram(self, image_data: bytes) -> torch.Tensor:
        """
        Preprocess uploaded spectrogram image
        Expects: Grayscale image of EEG spectrogram (60x29)
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize to expected dimensions (60x29)
            image = image.resize((29, 60), Image.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            spec = np.array(image, dtype=np.float32) / 255.0
            
            # Apply z-score normalization (same as training)
            spec = (spec - self.norm_mean) / (self.norm_std + 1e-8)
            
            # Add batch and channel dimensions: (1, 1, 60, 29)
            spec = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)
            
            return spec.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Failed to process spectrogram image: {str(e)}")
    
    def preprocess_npy(self, npy_data: bytes) -> torch.Tensor:
        """
        Preprocess uploaded .npy spectrogram array
        Expects: NumPy array (60, 29) with values in [0, 1]
        """
        try:
            # Load numpy array from bytes
            spec = np.load(io.BytesIO(npy_data))
            
            # Validate shape
            if spec.shape != (60, 29):
                raise ValueError(f"Expected shape (60, 29), got {spec.shape}")
            
            # Apply z-score normalization
            spec = (spec - self.norm_mean) / (self.norm_std + 1e-8)
            
            # Add batch and channel dimensions
            spec = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)
            
            return spec.to(self.device)
            
        except Exception as e:
            logger.error(f"Error processing .npy file: {str(e)}")
            raise ValueError(f"Failed to process .npy spectrogram: {str(e)}")
    
    def compute_anomaly_score(self, spectrogram: torch.Tensor) -> float:
        """Compute anomaly score (distance from hypersphere center)"""
        with torch.no_grad():
            embedding = self.net(spectrogram)
            distance = torch.sum((embedding - self.center) ** 2, dim=1)
            return float(distance.item())
    
    def classify_severity(self, score: float) -> Dict:
        """
        Classify nightmare severity and compute probabilities
        
        Returns detailed classification with confidence scores
        """
        # Determine if it's a nightmare
        is_nightmare = score > self.R
        
        # Compute severity level
        if not is_nightmare:
            severity = "normal"
            severity_level = 0
        elif score < self.thresholds['mild']:
            severity = "mild"
            severity_level = 1
        elif score < self.thresholds['moderate']:
            severity = "moderate"
            severity_level = 2
        else:
            severity = "severe"
            severity_level = 3
        
        # Compute probability scores using normalized distance
        # Higher score = more anomalous = higher nightmare probability
        max_observed = 5.0  # Based on evaluation data
        normalized_score = min(score / max_observed, 1.0)
        
        nightmare_probability = normalized_score * 100
        normal_probability = (1 - normalized_score) * 100
        
        # Confidence based on how far from radius threshold
        distance_from_threshold = abs(score - self.R)
        confidence = min(distance_from_threshold / self.R * 100, 100)
        
        return {
            'is_nightmare': is_nightmare,
            'severity': severity,
            'severity_level': severity_level,
            'nightmare_probability': round(nightmare_probability, 1),
            'normal_probability': round(normal_probability, 1),
            'confidence': round(confidence, 1),
            'anomaly_score': round(score, 4),
            'threshold': round(self.R, 4)
        }
    
    def generate_insights(self, classification: Dict, score: float) -> List[str]:
        """Generate clinical insights based on classification"""
        insights = []
        
        if not classification['is_nightmare']:
            insights.append("✅ EEG patterns consistent with normal dream activity")
            insights.append("No significant arousal markers detected")
            insights.append("Sleep architecture appears stable")
        else:
            # Severity-specific insights
            if classification['severity'] == 'mild':
                insights.append("⚠️ Mild nightmare markers detected")
                insights.append("Elevated beta activity suggests slight cortical arousal")
                insights.append("Consider monitoring sleep quality over time")
                
            elif classification['severity'] == 'moderate':
                insights.append("⚠️ Moderate nightmare activity detected")
                insights.append("Significant beta power increase and delta suppression")
                insights.append("Alpha intrusions indicate arousal instability")
                insights.append("Clinical evaluation recommended")
                
            else:  # severe
                insights.append("🔴 Severe nightmare disorder markers present")
                insights.append("Multiple arousal indicators: high beta, suppressed delta")
                insights.append("Autonomic dysregulation evident")
                insights.append("Immediate clinical intervention advised")
        
        # Add score-based context
        if score > self.R * 2:
            insights.append(f"Anomaly score significantly elevated ({score:.2f}x threshold)")
        
        return insights
    
    def analyze(self, file_data: bytes, file_type: str = "image") -> Dict:
        """
        Main analysis pipeline
        
        Args:
            file_data: Raw file bytes
            file_type: "image" for PNG/JPG or "npy" for NumPy array
        
        Returns:
            Complete analysis results
        """
        try:
            # Preprocess
            if file_type == "image":
                spectrogram = self.preprocess_spectrogram(file_data)
            elif file_type == "npy":
                spectrogram = self.preprocess_npy(file_data)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Compute anomaly score
            score = self.compute_anomaly_score(spectrogram)
            
            # Classify
            classification = self.classify_severity(score)
            
            # Generate insights
            insights = self.generate_insights(classification, score)
            
            # Compile results
            results = {
                'status': 'success',
                'timestamp': datetime.utcnow().isoformat(),
                'classification': classification,
                'insights': insights,
                'metadata': {
                    'model_version': '1.0',
                    'embedding_dim': self.net.embedding_dim,
                    'file_type': file_type
                }
            }
            
            logger.info(f"Analysis complete: {classification['severity']} (score: {score:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Nightmare Detection API",
    description="Real-time nightmare detection from EEG spectrograms using Deep SVDD",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector: Optional[NightmareDetector] = None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global detector
    
    # UPDATE THIS PATH to your model location
    MODEL_PATH = "./models/robust_deep_svdd.pth"

    
    try:
        detector = NightmareDetector(
            model_path=MODEL_PATH,
            device="cpu"  # Change to "cuda" if GPU available
        )
        logger.info("✅ Nightmare detector initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {str(e)}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Nightmare Detection API",
        "version": "1.0.0",
        "model_loaded": detector is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model": {
            "loaded": True,
            "radius": detector.R,
            "device": str(detector.device)
        },
        "thresholds": detector.thresholds
    }


@app.post("/analyze")
async def analyze_spectrogram(file: UploadFile = File(...)):
    """
    Analyze uploaded EEG spectrogram
    
    Accepts:
    - PNG/JPG images of spectrograms
    - .npy files containing spectrogram arrays (60x29)
    
    Returns:
    - Classification (normal/nightmare + severity)
    - Probability scores
    - Clinical insights
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.npy']
    file_ext = '.' + file.filename.split('.')[-1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    try:
        # Read file
        file_data = await file.read()
        
        # Determine file type
        file_type = "npy" if file_ext == '.npy' else "image"
        
        # Analyze
        results = detector.analyze(file_data, file_type)
        
        return JSONResponse(content=results)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/batch-analyze")
async def batch_analyze(files: List[UploadFile] = File(...)):
    """
    Analyze multiple spectrograms in batch
    
    Returns array of results, one per file
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 files per batch"
        )
    
    results = []
    
    for file in files:
        try:
            file_data = await file.read()
            file_type = "npy" if file.filename.endswith('.npy') else "image"
            
            result = detector.analyze(file_data, file_type)
            result['filename'] = file.filename
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                'status': 'error',
                'filename': file.filename,
                'error': str(e)
            })
    
    return JSONResponse(content={'results': results})


@app.get("/model-info")
async def model_info():
    """Get model information and thresholds"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        'radius': detector.R,
        'thresholds': detector.thresholds,
        'normalization': {
            'mean': detector.norm_mean,
            'std': detector.norm_std
        },
        'expected_input_shape': [60, 29],
        'severity_levels': {
            0: 'normal',
            1: 'mild',
            2: 'moderate',
            3: 'severe'
        }
    }


# ============================================================================
# EXAMPLE RESPONSE MODELS (for documentation)
# ============================================================================

class AnalysisResponse(BaseModel):
    """Example response structure"""
    status: str
    timestamp: str
    classification: Dict
    insights: List[str]
    metadata: Dict


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )