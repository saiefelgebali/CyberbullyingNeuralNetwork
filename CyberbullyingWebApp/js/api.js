const API = "http://localhost:5224";

export const predictText = async (text) => {
	const res = await fetch(
		`${API}/cyberbullying?` +
			new URLSearchParams({
				text,
			}),
		{
			method: "GET",
			headers: {
				"Content-Type": "application/json",
				Accept: "application/json",
				"Cors-Allow-Origin": "*",
			},
			params: {
				text: text,
			},
		}
	);

	const json = await res.json();

	return json;
};
