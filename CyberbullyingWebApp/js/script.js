import { predictText } from "./api.js";

let currentPage = document.getElementById("home");
const input = document.getElementById("input");

// Ctrl + Enter to submit
input.addEventListener("keypress", (e) => {
	if (e.key === "\n" && e.ctrlKey) {
		handleResult();
	}
});

// submit form on click button
document.getElementById("submit").addEventListener("click", (e) => {
	handleResult();
});

// link back to home page
document.querySelectorAll(".return-to-home").forEach((element) => {
	element.addEventListener("click", (e) => {
		e.preventDefault();
		changePage("home");
	});
});

export async function handleResult() {
	const text = input.value;

	// make call to cyberbullying api to predict text
	const pred = await predictText(text);

	handlePred(pred);

	input.value = "";
}

// get class of prediction
function calculateResult(result) {
	if (result > 85) return "high";
	else if (result > 60) return "medium";
	else if (result > 0) return "low";
	else return "error";
}

// show prediction on screen
function handlePred(pred) {
	pred *= 100;
	const resultClass = calculateResult(pred);
	changePage(`result-${resultClass}`);

	const result = pred > 0 ? `${Math.floor(pred)}%` : "N/A";

	document.querySelector(`.result.${resultClass}`).innerHTML = result;
}

function changePage(pageId) {
	// Hide old page
	currentPage.classList.add("hidden");

	// Show new page
	const page = document.getElementById(pageId);
	page.classList.remove("hidden");
	currentPage = page;
}
