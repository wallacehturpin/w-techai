
class MyAI {
    constructor() {
        this.inputs = 8;
        this.hidden = 12;
        this.outputs = 4;

        this.weights = Array.from(
            { length: this.inputs * this.hidden },
            () => Math.random() * 0.2 - 0.1
        );

        this.outputWeights = Array.from(
            { length: this.hidden * this.outputs },
            () => Math.random() * 0.2 - 0.1
        );

        this.bias = Array(this.hidden).fill(0);
        this.outputBias = Array(this.outputs).fill(0);

        this.training = [
            { words: ["hello", "hi", "hey"], answer: 0 },
            { words: ["bye", "goodbye"], answer: 1 },
            { words: ["thanks", "thank"], answer: 2 },
            { words: ["name", "who"], answer: 3 }
        ];

        this.responses = [
            "Hello! Nice to meet you!",
            "Goodbye! Have a great day!",
            "You're welcome!",
            "I'm your own AI chatbot!"
        ];

        this.train();
    }

    encode(message) {
        const words = message.toLowerCase().match(/[a-z]+/g) || [];
        const features = Array(this.inputs).fill(0);

        words.forEach(word => {
            features[this.hash(word) % this.inputs] = 1;
        });

        return features;
    }

    hash(word) {
        let value = 0;

        for (const char of word) {
            value = (value * 31 + char.charCodeAt(0)) >>> 0;
        }

        return value;
    }

    sigmoid(value) {
        return 1 / (1 + Math.exp(-value));
    }

    forward(input) {
        const hidden = [];

        for (let j = 0; j < this.hidden; j++) {
            let sum = this.bias[j];

            for (let i = 0; i < this.inputs; i++) {
                sum += input[i] *
                    this.weights[i * this.hidden + j];
            }

            hidden[j] = this.sigmoid(sum);
        }

        const output = [];

        for (let k = 0; k < this.outputs; k++) {
            let sum = this.outputBias[k];

            for (let j = 0; j < this.hidden; j++) {
                sum += hidden[j] *
                    this.outputWeights[j * this.outputs + k];
            }

            output[k] = this.sigmoid(sum);
        }

        return { hidden, output };
    }

    train() {
        const rate = 0.3;

        for (let epoch = 0; epoch < 1000; epoch++) {
            for (const example of this.training) {
                const input = this.encode(example.words.join(" "));
                const { hidden, output } = this.forward(input);

                const targets = Array(this.outputs).fill(0);
                targets[example.answer] = 1;

                const outputErrors = output.map(
                    (value, i) => targets[i] - value
                );

                const hiddenErrors = Array(this.hidden).fill(0);

                for (let j = 0; j < this.hidden; j++) {
                    for (let k = 0; k < this.outputs; k++) {
                        hiddenErrors[j] +=
                            outputErrors[k] *
                            this.outputWeights[j * this.outputs + k];
                    }
                }

                for (let j = 0; j < this.hidden; j++) {
                    for (let k = 0; k < this.outputs; k++) {
                        const index = j * this.outputs + k;

                        this.outputWeights[index] +=
                            rate * outputErrors[k] *
                            hidden[j] * output[k] * (1 - output[k]);
                    }
                }

                for (let k = 0; k < this.outputs; k++) {
                    this.outputBias[k] +=
                        rate * outputErrors[k] *
                        output[k] * (1 - output[k]);
                }

                for (let i = 0; i < this.inputs; i++) {
                    for (let j = 0; j < this.hidden; j++) {
                        const index = i * this.hidden + j;

                        this.weights[index] +=
                            rate * hiddenErrors[j] *
                            hidden[j] * (1 - hidden[j]) * input[i];
                    }
                }

                for (let j = 0; j < this.hidden; j++) {
                    this.bias[j] +=
                        rate * hiddenErrors[j] *
                        hidden[j] * (1 - hidden[j]);
                }
            }
        }
    }

    answer(message) {
        const { output } = this.forward(this.encode(message));
        const best = output.indexOf(Math.max(...output));

        return Math.max(...output) > 0.5
            ? this.responses[best]
            : "I'm still learning. Can you teach me something?";
    }
}

const ai = new MyAI();
