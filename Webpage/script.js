document.addEventListener("DOMContentLoaded", async function () {
    const container = document.getElementById("champion-grid");
    const droppableBoxes = document.querySelectorAll(".box.dropeable");

    try {
        const response = await fetch("assets/champions_ids.json");
        const champions = await response.json();

        let html = "<table style='margin: auto;'><tbody>";
        let count = 0;

        Object.keys(champions).forEach(champion => {
            if (count % 6 === 0) html += "<tr>";

            const imgElement = `<td style='text-align: center; padding: 15px;'>
                <div style="display: flex; align-items: center; justify-content: center; width: 7rem; height: 7rem; background-image: url('assets/champion_images/${champion}.png');
                background-size: cover; background-position: center; border-radius: 30px; cursor: grab; margin: auto;"
                draggable='true' data-champion='${champion}'></div>
                <span style='display: block; font-size: 14px; font-weight: bold; color: white; margin-top: 5px;'>${champion}</span>
            </td>`;

            html += imgElement;
            count++;

            if (count % 6 === 0) html += "</tr>";
        });

        if (count % 6 !== 0) html += "</tr>";
        html += "</tbody></table>";

        container.innerHTML = html;

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
                        }
                    }

                    box.style.backgroundImage = `url('assets/champion_images/${newChampion}.png')`;
                    box.setAttribute("champion", newChampion);

                    const newChampionElement = document.querySelector(`[data-champion='${newChampion}']`);
                    if (newChampionElement) {
                        newChampionElement.setAttribute("draggable", "false");
                        newChampionElement.style.cursor = "not-allowed";
                        newChampionElement.style.opacity = "0.5"; 
                        newChampionElement.style.filter = "grayscale(100%)"
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
                                newChampionElement.style.filter = "grayscale(0%)"
                            }
                        });

                        box.style.position = "relative";
                        box.appendChild(removeBtn);
                    }
                }
            });
        });

    } catch (error) {
        console.error("Error loading champions JSON:", error);
    }
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
    const winnerOverlay = document.getElementById("winner-overlay");
    const container = document.getElementById("champion-grid");

    winnerOverlay.style.display = "flex";
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
    
    console.log("Blue Side Players:", JSON.stringify(bluePlayers, null, 2));
    console.log("Blue Side Champions:", JSON.stringify(blueChampions, null, 2));
    console.log("Red Side Players:", JSON.stringify(redPlayers, null, 2));
    console.log("Red Side Champions:", JSON.stringify(redChampions, null, 2));
});

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