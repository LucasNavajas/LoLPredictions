import { API_URL } from './config.js';
document.addEventListener("DOMContentLoaded", async function () {
    const container = document.getElementById("champion-grid");
    const searchInput = document.getElementById("champion-search");

    container.parentNode.insertBefore(searchInput, container);

    const droppableBoxes = document.querySelectorAll(".box.dropeable");
    let championsData = {};

    async function loadChampions() {
        try {
            const response = await fetch("assets/champions_ids.json");
            championsData = await response.json();
            renderChampions();
        } catch (error) {
            console.error("Error loading champions JSON:", error);
        }
    }

    function renderChampions(filter = "") {
    let html = `<div style="height: 620px; overflow: auto; display: flex; justify-content: center;">
                    <table style="margin: auto; height: 100%;"><tbody>`;

    let count = 0;
    let filteredChampions = Object.keys(championsData).filter(champion =>
        !filter || champion.toLowerCase().includes(filter.toLowerCase())
    );

    const minColumns = 6;
    const minRows = 3; 
    const minElements = minColumns * minRows;


    if (filteredChampions.length < minElements) {
        while (filteredChampions.length < minElements) {
            filteredChampions.push(null); 
        }
    }

    filteredChampions.forEach((champion, index) => {
        if (index % minColumns === 0) html += "<tr>";

        if (champion) {
            html += `<td style="text-align: center; padding: 15px;">
                <div style="display: flex; align-items: center; justify-content: center; width: 7rem; height: 7rem; background-image: url('assets/champion_images/${champion}.png');
                background-size: cover; background-position: center; border-radius: 30px; cursor: grab; margin: auto;"
                draggable="true" data-champion="${champion}"></div>
                <span style="display: block; font-size: 14px; font-weight: bold; color: white; margin-top: 5px;">${champion}</span>
            </td>`;
        } else {
            html += "<td style='text-align: center; padding: 15px;'></td>"; // Empty cell placeholder
        }

        if ((index + 1) % minColumns === 0) html += "</tr>";
    });

    html += "</tbody></table></div>";

    container.innerHTML = html;
    addDragAndDropEvents();
}


    function addDragAndDropEvents() {
        document.querySelectorAll("[draggable='true']").forEach(imgElement => {
            imgElement.addEventListener("dragstart", function (event) {
                event.dataTransfer.setData("text", imgElement.dataset.champion);
            });
        });

        droppableBoxes.forEach(box => {
            box.addEventListener("dragover", function (event) {
                event.preventDefault();
            });

            box.addEventListener("drop", function (event) {
                event.preventDefault();
                const newChampion = event.dataTransfer.getData("text");

                if (newChampion) {
                    const previousChampion = box.getAttribute("champion");

                    if (previousChampion) {
                        const previousChampionElement = document.querySelector(`[data-champion='${previousChampion}']`);
                        if (previousChampionElement) {
                            previousChampionElement.setAttribute("draggable", "true");
                            previousChampionElement.style.cursor = "grab";
                            previousChampionElement.style.opacity = "1";
                            previousChampionElement.style.filter = "grayscale(0%)";
                        }
                    }

                    box.style.backgroundImage = `url('assets/champion_images/${newChampion}.png')`;
                    box.setAttribute("champion", newChampion);

                    const newChampionElement = document.querySelector(`[data-champion='${newChampion}']`);
                    if (newChampionElement) {
                        newChampionElement.setAttribute("draggable", "false");
                        newChampionElement.style.cursor = "not-allowed";
                        newChampionElement.style.opacity = "0.5"; 
                        newChampionElement.style.filter = "grayscale(100%)";
                    }

                    if (!box.querySelector(".remove-button")) {
                        const removeBtn = document.createElement("img");
                        removeBtn.src = "assets/imgs/x.png";
                        removeBtn.classList.add("remove-button");
                        removeBtn.style.backgroundColor = "rgba(0, 0, 0, 0.85)";
                        removeBtn.style.position = "absolute";
                        removeBtn.style.top = "5px";
                        removeBtn.style.right = "5px";
                        removeBtn.style.width = "25px";
                        removeBtn.style.height = "25px";
                        removeBtn.style.cursor = "pointer";
                        removeBtn.style.borderRadius = "50%";
                        removeBtn.style.padding = "3px";

                        removeBtn.addEventListener("click", function () {
                            box.removeAttribute("champion");
                            box.style.backgroundImage = "url(assets/champion_images/-1.png)";
                            removeBtn.remove();

                            if (newChampionElement) {
                                newChampionElement.setAttribute("draggable", "true");
                                newChampionElement.style.cursor = "grab";
                                newChampionElement.style.opacity = "1";
                                newChampionElement.style.filter = "grayscale(0%)";
                            }
                        });

                        box.style.position = "relative";
                        box.appendChild(removeBtn);
                    }
                }
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
document.getElementById("predict-button").addEventListener("click", function () {
    processPrediction(false);
});

document.getElementById("predict-draft-button").addEventListener("click", function () {
    processPrediction(true); 
});

function processPrediction(isDraft) {
    
    const winnerOverlay = document.getElementById("winner-overlay");
    const container = document.getElementById("champion-grid");
    const loadingImage = document.getElementById("loading-image");
    
    loadingImage.style.display = "block";
    winnerOverlay.style.display = "none";
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
            bluePlayers.push(isDraft ? "" : input.value);
            blueChampions.push(championName);
        } else {
            redPlayers.push(isDraft ? "" : input.value);
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

    fetch(API_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        console.log("API Response:", data);

        loadingImage.style.display = "none";
        winnerOverlay.style.display = "flex";
        winnerOverlay.style.flexDirection = "column";

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
        loadingImage.style.display = "none";
        container.style.visibility = "visible";
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
        } 
        else if (allBoxesHaveChampion) {
            predictButton.disabled = true;
            predictButton.classList.remove("enabled");

            predictDraftButton.disabled = false;
            predictDraftButton.classList.add("enabled");
            
        }
        else {
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