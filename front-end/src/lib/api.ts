const API_BASE = 'http://localhost:8000/api'; // Update this for production

export async function classifyExoplanetFile(file: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/classify/`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error?.error || 'Failed to classify file');
  }

  const result = await response.json();
  return result.results;
}