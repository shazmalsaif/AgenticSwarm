// Production backend API URL on Render (NO trailing slash).
// Automatically stays empty ('') on localhost / 127.0.0.1 so local dev uses local FastAPI,
// and routes deployed Vercel (or any other remote frontend origin) to Render.
(function () {
  const host = (window.location && window.location.hostname) || '';
  const isLocal = !host || host === 'localhost' || host === '127.0.0.1';
  window.AGENTSWARM_API_URL = isLocal ? '' : 'https://agenticswarm.onrender.com';
})();
