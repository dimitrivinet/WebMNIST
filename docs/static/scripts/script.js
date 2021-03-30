
var content = document.getElementById("content");

var digits = [];

for (let i = 0; i < 10; i++)
{
    let digit = {
        container: document.createElement("div"),
        content: {
            bar: document.createElement("div"),
            label: document.createElement("div"),
        }
    }

    digit.container.id = i;
    digit.content.bar.id = "bar" + i;
    digit.content.label.id = "label" + i;

    digit.content.bar.classList.add("bar");
    digit.content.label.classList.add("label");

    content.appendChild(digit.container)
    digit.container.appendChild(digit.content.bar)
    digit.container.appendChild(digit.content.label)

    digit.content.label.innerHTML = i;

    digits.push(digit);
}