document.addEventListener("DOMContentLoaded", async function () {
    const container = document.getElementById("champion-grid");
    const searchInput = document.getElementById("champion-search");
    searchInput.setAttribute("type", "text");
    searchInput.setAttribute("id", "champion-search");
    searchInput.setAttribute("placeholder", "Search Champion...");

    container.parentNode.insertBefore(searchInput, container);

    let championsData = {};
    let allChampions = [];

    async function loadChampions() {
        try {
            const response = await fetch("assets/champions_ids.json");
            championsData = await response.json();
            allChampions = Object.keys(championsData);
            renderChampions();
        } catch (error) {
            console.error("Error loading champions JSON:", error);
        }
    }

    function renderChampions(filter = "") {
        let filteredChampions = allChampions.filter(champion =>
            champion.toLowerCase().includes(filter.toLowerCase())
        );

        let minRows = 3;
        let minCells = minRows * 6;
        let totalCells = Math.max(minCells, Math.ceil(filteredChampions.length / 6) * 6);

        let html = "<table style='margin: auto; height: 300px;'><tbody>";
        let count = 0;

        for (let i = 0; i < totalCells; i++) {
            if (count % 6 === 0) html += "<tr>";

            if (i < filteredChampions.length) {
                const champion = filteredChampions[i];
                html += `<td style='text-align: center; padding: 15px;'>
                    <div style="display: flex; align-items: center; justify-content: center; width: 7rem; height: 7rem; background-image: url('assets/champion_images/${champion}.png');
                    background-size: cover; background-position: center; border-radius: 30px; cursor: grab; margin: auto;"
                    draggable='true' data-champion='${champion}'></div>
                    <span style='display: block; font-size: 14px; font-weight: bold; color: white; margin-top: 5px;'>${champion}</span>
                </td>`;
            } else {
                // Add empty placeholders to maintain height
                html += `<td style="text-align: center; padding: 15px; opacity: 0;">
                    <div style="width: 7rem; height: 7rem;"></div>
                </td>`;
            }

            count++;
            if (count % 6 === 0) html += "</tr>";
        }

        html += "</tbody></table>";

        container.innerHTML = html;
        addDragAndDropEvents();
    }

    function addDragAndDropEvents() {
        document.querySelectorAll("[draggable='true']").forEach(imgElement => {
            imgElement.addEventListener("dragstart", function (event) {
                event.dataTransfer.setData("text", imgElement.dataset.champion);
            });
        });
    }

    searchInput.addEventListener("input", function () {
        renderChampions(searchInput.value);
    });

    loadChampions();
});




document.addEventListener("DOMContentLoaded", async () => {
    let playerNames = [];
    let playerNamesLower = [];
    try {
        const response = await fetch("assets/players_ids.json");
        const data = await response.json();
        playerNamesLower = Object.keys(data).map(name => name.toLowerCase());
        playerNames = Object.keys(data)
    } catch (error) {
        console.error("Error loading player data:", error);
    }

    const inputs = document.querySelectorAll(".box-label");

    inputs.forEach((input, index) => {
        input.setAttribute("autocomplete", "off");

        const dataList = document.createElement("datalist");
        dataList.id = `datalist-${input.dataset.slot}`;
        document.body.appendChild(dataList);
        input.setAttribute("list", dataList.id);
        const errorMessage = input.nextElementSibling; 

        input.addEventListener("focus", () => updateDatalist(dataList, input.value));
        input.addEventListener("input", () => updateDatalist(dataList, input.value));

        input.addEventListener("change", function () {
            if (!playerNamesLower.includes(this.value.toLowerCase())) {

                this.style.border = "2px solid red";
                errorMessage.style.visibility = "visible"

                const predictButton = document.getElementById("predict-button");
                const predictDraftButton = document.getElementById("predict-draft-button");
                predictButton.disabled = true;
                predictButton.classList.remove("enabled");
                predictDraftButton.disabled = true;
                predictDraftButton.classList.remove("enabled");
                this.value = "";
            } else {
                this.style.border = "2px solid white";
                errorMessage.style.visibility = "hidden"
                dataList.remove();
                moveToNextInput(index);
                
            }
        });

        input.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
                event.preventDefault();
                dataList.remove();
                moveToNextInput(index);
            }
        });
        dataList.style.display = "none";
    });

    function updateDatalist(dataList, inputValue) {
        dataList.innerHTML = "";
        if (inputValue.length < 1) return;
        const filteredNames = playerNames.filter(name => name.toLowerCase().includes(inputValue.toLowerCase()));
        
        filteredNames.slice(0, 4).forEach(name => {
            const option = document.createElement("option");
            option.value = name;
            dataList.appendChild(option);
        });
        
    }

    function moveToNextInput(index) {
        const nextInput = inputs[index + 1];
        if (nextInput) {
            nextInput.focus();
        }
    }
});
document.getElementById("predict-button").addEventListener("click", function() {
    processPrediction();
});

document.getElementById("predict-draft-button").addEventListener("click", function() {
    processPrediction();
});

function processPrediction() {
    const winnerOverlay = document.getElementById("winner-overlay");
    const container = document.getElementById("champion-grid");

    winnerOverlay.style.display = "flex";
    winnerOverlay.style.flexDirection = "column";
    container.style.visibility = "hidden";

    let bluePlayers = [];
    let blueChampions = [];
    let redPlayers = [];
    let redChampions = [];

    document.querySelectorAll(".box-label").forEach(input => {
        let slot = parseInt(input.getAttribute("data-slot"));
        let championBox = document.querySelector(`.box[data-slot='${slot}']`);
        let bgImage = championBox.style.backgroundImage;
        let championName = bgImage.match(/assets\/champion_images\/(.*?).png/);
        championName = championName ? championName[1] : "Unknown";

        if (slot >= 1 && slot <= 5) {
            bluePlayers.push(input.value);
            blueChampions.push(championName);
        } else {
            redPlayers.push(input.value);
            redChampions.push(championName);
        }
    });

    let requestData = {
        players1: bluePlayers,
        players2: redPlayers,
        champions1: blueChampions,
        champions2: redChampions
    };

    console.log("Sending Request Data:", JSON.stringify(requestData, null, 2));

    fetch("https://r3zwjykyn7.execute-api.us-east-2.amazonaws.com/LolPredictions", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        console.log("API Response:", data);

        const prediction = data.prediction;
        const chanceText = document.querySelector("#winner-overlay .chance-text");
        const winnerText = document.querySelector("#winner-overlay div:nth-child(2)");

        if (prediction < 0.5) {
            winnerText.textContent = "Blue Team Wins";
            chanceText.textContent = `${((1 - prediction) * 100).toFixed(2)}% chance`;
        } else {
            winnerText.textContent = "Red Team Wins";
            chanceText.textContent = `${(prediction * 100).toFixed(2)}% chance`;
        }
    })
    .catch(error => {
        console.error("Error calling API:", error);
    });
}



document.addEventListener("DOMContentLoaded", function () {
    const predictButton = document.getElementById("predict-button");
    const predictDraftButton = document.getElementById("predict-draft-button");
    const inputs = document.querySelectorAll(".box-label");
    const boxes = document.querySelectorAll(".box.dropeable");
    const closeOverlay = document.getElementById("close-overlay");

    function checkConditions() {
        let allInputsFilled = Array.from(inputs).every(input => input.value.trim() !== "");
        let allBoxesHaveChampion = Array.from(boxes).every(box => box.hasAttribute("champion"));

        if (allInputsFilled && allBoxesHaveChampion) {
            predictButton.disabled = false;
            predictButton.classList.add("enabled");

            predictDraftButton.disabled = false;
            predictDraftButton.classList.add("enabled");
        } else {
            predictButton.disabled = true;
            predictButton.classList.remove("enabled");

            predictDraftButton.disabled = true;
            predictDraftButton.classList.remove("enabled");
        }
    }

    inputs.forEach(input => {
        input.addEventListener("input", checkConditions);
    });

    const observer = new MutationObserver(() => checkConditions());

    boxes.forEach(box => {
        observer.observe(box, { attributes: true, attributeFilter: ["champion"] });
    });

    checkConditions();

    function closeWinnerOverlay() {
        const container = document.getElementById("champion-grid");
        const winnerOverlay = document.getElementById("winner-overlay");
        winnerOverlay.style.display = "none";
        container.style.visibility = "visible"
    }

    closeOverlay.addEventListener("click", closeWinnerOverlay);
});