// JavaScript to handle form submission and dynamically display results
document.getElementById("searchForm").addEventListener("submit", async (event) => {
    event.preventDefault(); // Prevent the default form submission

    const formData = new FormData(event.target);
    const resultsContainer = document.getElementById("resultsContainer");
    const errorMessage = document.getElementById("errorMessage");

    // Clear previous results and error message
    resultsContainer.innerHTML = "<h2>Search Results</h2>";
    errorMessage.textContent = "";

    try {
        // Send POST request to the server
        const response = await fetch("/", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to fetch results from the server.");
        }

        const results = await response.json();

        if (results.length === 0) {
            resultsContainer.innerHTML += "<p>No results found.</p>";
            return;
        }

        // Display search results
        results.forEach((result) => {
            const resultItem = document.createElement("div");
            resultItem.classList.add("result-item");

            const img = document.createElement("img");
            img.src = `/coco_images_resized/${result[0]}`; // Use the correct directory
            img.alt = "Result Image";

            const similarity = document.createElement("p");
            similarity.textContent = `Similarity Score: ${result[1].toFixed(3)}`;

            resultItem.appendChild(img);
            resultItem.appendChild(similarity);
            resultsContainer.appendChild(resultItem);
        });
    } catch (error) {
        console.error("Error during search:", error);
        errorMessage.textContent = error.message;
    }
});
