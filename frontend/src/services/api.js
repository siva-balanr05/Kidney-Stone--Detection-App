import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
});

// Response interceptor for unified error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const msg = error.response.data?.detail || `Server error: ${error.response.status}`;
      return Promise.reject(new Error(msg));
    } else if (error.request) {
      return Promise.reject(new Error('Cannot connect to the KidneyAI API. Make sure the backend is running on port 8000.'));
    }
    return Promise.reject(error);
  }
);

const apiService = {
  // Health check
  async getHealth() {
    const res = await api.get('/api/health');
    return res.data;
  },

  // Predict from image file
  async predictImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post('/api/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // Get analysis history (paginated)
  async getHistory({ skip = 0, limit = 20, prediction = null } = {}) {
    const params = { skip, limit };
    if (prediction) params.prediction = prediction;
    const res = await api.get('/api/history', { params });
    return res.data;
  },

  // Get a single analysis by ID
  async getAnalysis(analysisId) {
    const res = await api.get(`/api/history/${analysisId}`);
    return res.data;
  },

  // Delete an analysis
  async deleteAnalysis(analysisId) {
    const res = await api.delete(`/api/history/${analysisId}`);
    return res.data;
  },

  // Get model info
  async getModelInfo() {
    const res = await api.get('/api/model-info');
    return res.data;
  },

  // Get dashboard stats
  async getStats() {
    const res = await api.get('/api/stats');
    return res.data;
  },
};

export default apiService;
