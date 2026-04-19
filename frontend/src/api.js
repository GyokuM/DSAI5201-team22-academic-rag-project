export async function fetchJson(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json();
}

export const api = {
  health: () => fetchJson("/api/health"),
  dashboard: () => fetchJson("/api/dashboard"),
  papers: (query = "") =>
    fetchJson(`/api/papers${query ? `?query=${encodeURIComponent(query)}` : ""}`),
  paper: (paperId) => fetchJson(`/api/papers/${paperId}`),
  paperQuestions: (paperId) => fetchJson(`/api/papers/${paperId}/questions`),
  askPaper: (paperId, payload) =>
    fetchJson(`/api/papers/${paperId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  uploadPdf: async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch("/api/uploads/pdf", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || `Upload failed: ${response.status}`);
    }
    return response.json();
  },
  askUploaded: (uploadId, payload) =>
    fetchJson(`/api/uploads/${uploadId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
};
