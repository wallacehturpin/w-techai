
const chat = document.getElementById("chat");
const input = document.getElementById("userInput");
const button = document.getElementById("sendButton");

function addMessage(message, type) {
    const div = document.createElement("div");

    div.className = "message " + type;
    div.textContent = message;

    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function sendMessage() {
    const message = input.value.trim();

    if (!message) return;

    addMessage(message, "user");
    input.value = "";

    const response = ai.answer(message);

    addMessage(response, "ai");
}

button.addEventListener("click", sendMessage);

input.addEventListener("keydown", event => {
    if (event.key === "Enter") {
        sendMessage();
    }
});
