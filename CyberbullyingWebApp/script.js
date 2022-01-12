let currentPage = document.getElementById("home");

function handleResult(result) {
	const resultClass = calculateResult(result);
	changePage(`result-${resultClass}`);
}

function calculateResult(result) {
	if (result > 70) return "high";
	else if (result > 30) return "medium";
	else if (result > 0) return "low";
	else return "error";
}

function changePage(pageId) {
	// Hide old page
	currentPage.classList.add("hidden");

	// Show new page
	const page = document.getElementById(pageId);
	page.classList.remove("hidden");
	currentPage = page;
}
