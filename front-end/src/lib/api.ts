const API_BASE = 'http://localhost:8000/api'; // Update this for production

export async function classifyExoplanet(data: {
  radius: string;
  mass: string;
  orbital_period: string;
  temperature: string;
}) {
  const response = await fetch(`${API_BASE}/classify/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });

  if (!response.ok) {
    throw new Error('API request failed');
  }

  const result = await response.json();
  return result.classification;
}