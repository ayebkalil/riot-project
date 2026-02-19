
import React from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import AnalyticsDashboard from './pages/AnalyticsDashboard';
import ModelDashboard from './pages/ModelDashboard';
import Predictions from './pages/Predictions';
import Profile from './pages/Profile';

const App: React.FC = () => {
  return (
    <HashRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<AnalyticsDashboard />} />
          <Route path="/models" element={<ModelDashboard />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Layout>
    </HashRouter>
  );
};

export default App;
