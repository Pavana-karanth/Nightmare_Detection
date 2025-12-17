// Global API configuration
// Replace with your ngrok URL when running the backend
export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export const API_ENDPOINTS = {
  ANALYZE: `${API_URL}/analyze`,
  HEALTH: `${API_URL}/health`,
} as const
