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
        let selectedChampions = new Set();
        droppableBoxes.forEach(box => {
            const champion = box.getAttribute("champion");
            if (champion) selectedChampions.add(champion);
        });
    
        let html = `<div style="height: 620px; overflow: auto; display: flex; justify-content: center;">
                        <table style="margin: auto; height: 100%;"><tbody>`;
    
        let count = 0;
        let filteredChampions = Object.keys(championsData).filter(champion =>
            !filter || champion.toLowerCase().includes(filter.toLowerCase()) // Keep all champions visible
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
                const isDisabled = selectedChampions.has(champion);
                html += `<td style="text-align: center; padding: 15px;">
                    <div style="display: flex; align-items: center; justify-content: center; width: 7rem; height: 7rem; background-image: url('assets/champion_images/${champion}.png');
                    background-size: cover; background-position: center; border-radius: 30px; cursor: ${isDisabled ? "not-allowed" : "grab"}; 
                    margin: auto; opacity: ${isDisabled ? "0.5" : "1"}; filter: ${isDisabled ? "grayscale(100%)" : "grayscale(0%)"};"
                    draggable="${isDisabled ? "false" : "true"}" data-champion="${champion}"></div>
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
                dataList.innerHTML = ""
                moveToNextInput(index);
                
            }
        });

        input.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
                event.preventDefault();
                dataList.innerHTML = ""
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

async function processPrediction(isDraft) {
    
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

    try {
        const originalResponse = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestData)
        });

        const originalData = await originalResponse.json();
        let originalProb = originalData.body.prediction;
        console.log("Original Prediction:", originalProb);

        if (isDraft) {
            let swappedRequestData = {
                players1: bluePlayers,
                players2: redPlayers,
                champions1: redChampions,
                champions2: blueChampions
            };

            console.log("Sending Swapped Request Data:", JSON.stringify(swappedRequestData, null, 2));

            const swappedResponse = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(swappedRequestData)
            });

            const swappedData = await swappedResponse.json();
            let swappedProb = swappedData.body.prediction;
            console.log("Swapped Prediction:", swappedProb);

            let { W_C_R, W_C_B } = calculateCompositionWinrates(originalProb, swappedProb);
            let finalWinner;
            if (originalProb < 0.5) {
                if (swappedProb > originalProb) {
                    finalWinner = "Blue Team Wins";
                }
                else {
                    finalWinner = "Red Team Wins"
                }
            }
            else {
                if (swappedProb < originalProb) {
                    finalWinner = "Red Team Wins";
                }
                else {
                    finalWinner = "Blue Team Wins"
                }
            }
            let finalChance = (Math.max(W_C_R, W_C_B) * 100).toFixed(2) + "% chance";

            displayPrediction(finalWinner, finalChance);
        } else {
            let finalWinner = originalProb < 0.5 ? "Blue Team Wins" : "Red Team Wins";
            let finalChance = ((originalProb < 0.5 ? 1 - originalProb : originalProb) * 100).toFixed(2) + "% chance";
            displayPrediction(finalWinner, finalChance);
        }
    } catch (error) {
        console.error("Error calling API:", error);
        loadingImage.style.display = "none";
        container.style.visibility = "visible";
    }


}

function calculateCompositionWinrates(originalProb, swappedProb) {
    let ratio1; 
    if (originalProb > 0.5) {
        ratio1 = swappedProb / originalProb
    }
    else {
        ratio1 = (1 - swappedProb) / (1 - originalProb);
    }

    let W_C_R = 1 / (1 + ratio1);
    let W_C_B = ratio1 * W_C_R;

    return { W_C_R, W_C_B };
}

function displayPrediction(winnerText, chanceText) {
    const winnerOverlay = document.getElementById("winner-overlay");
    const loadingImage = document.getElementById("loading-image");

    loadingImage.style.display = "none";
    winnerOverlay.style.display = "flex";
    winnerOverlay.style.flexDirection = "column";

    document.querySelector("#winner-overlay div:nth-child(2)").textContent = winnerText;
    document.querySelector("#winner-overlay .chance-text").textContent = chanceText;
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

document.getElementById('reset-champions').addEventListener('click', () => {
    document.querySelectorAll('.box').forEach(box => {
        box.style.backgroundImage = "url('assets/champion_images/-1.png')";

        let removeBtn = box.querySelector('.remove-button'); 
        if (removeBtn) {
            removeBtn.remove();
        }

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
        box.removeAttribute("champion");
    });
});

document.getElementById('reset-players').addEventListener('click', function() {
    document.querySelectorAll('.box-label').forEach(input => {
        input.value = '';
    });
});

document.getElementById('swap-players').addEventListener('click', function() {
    for (let i = 1; i <= 5; i++) {
        let blueInput = document.querySelector(`.box-label[data-slot="${i}"]`);
        let redInput = document.querySelector(`.box-label[data-slot="${i + 5}"]`);

        let temp = blueInput.value;
        blueInput.value = redInput.value;
        redInput.value = temp;
    }
});

document.getElementById('swap-champions').addEventListener('click', function() {
    for (let i = 1; i <= 5; i++) {
        let blueBox = document.querySelector(`.box.dropeable[data-slot="${i}"]`);
        let redBox = document.querySelector(`.box.dropeable[data-slot="${i + 5}"]`);

        if (blueBox && redBox) {
            let tempBackground = blueBox.style.backgroundImage;
            blueBox.style.backgroundImage = redBox.style.backgroundImage;
            redBox.style.backgroundImage = tempBackground;

            let tempChampion = blueBox.getAttribute("champion");
            blueBox.setAttribute("champion", redBox.getAttribute("champion"));
            redBox.setAttribute("champion", tempChampion);

            let blueRemoveBtn = blueBox.querySelector(".remove-button");
            let redRemoveBtn = redBox.querySelector(".remove-button");

            if (blueRemoveBtn) {
                blueBox.removeChild(blueRemoveBtn);
            }
            if (redRemoveBtn) {
                redBox.removeChild(redRemoveBtn);
            }

            if (blueBox.getAttribute("champion") && blueBox.getAttribute("champion") !== "null") {
                addRemoveButton(blueBox);
            }
            if (redBox.getAttribute("champion") && redBox.getAttribute("champion") !== "null") {
                addRemoveButton(redBox);
            }
        }
    }
});


function addRemoveButton(box) {
    if (!box.getAttribute("champion") || box.getAttribute("champion") === "null") {
        return;
    }

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
        let championName = box.getAttribute("champion");
        box.removeAttribute("champion");
        box.style.backgroundImage = "url(assets/champion_images/-1.png)";
        removeBtn.remove();

        let championElement = document.querySelector(`[data-champion='${championName}']`);
        if (championElement) {
            championElement.setAttribute("draggable", "true");
            championElement.style.cursor = "grab";
            championElement.style.opacity = "1";
            championElement.style.filter = "grayscale(0%)";
        }
    });

    box.style.position = "relative";
    box.appendChild(removeBtn);
}
