

var content = document.getElementById("content");


function preventDefault(e) {
    e = e || window.event;
    if (e.preventDefault)
        e.preventDefault();
    e.returnValue = false;  
  }
document.addEventListener('touchmove',preventDefault, false);

var digits = [];

for (let i = 0; i < 10; i++) {
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


var canvas = document.getElementById('canvas');
context = canvas.getContext("2d");
context.strokeStyle = "#ffffff";
context.lineJoin = "round";
context.lineCap = 'round';
context.lineWidth = 20;

var clickX = [];
var clickY = [];
var clickDrag = [];
var paint;

/**
 * Add information where the user clicked at.
 * @param {number} x
 * @param {number} y
 * @return {boolean} dragging
 */
function addClick(x, y, dragging) {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
}


/**
 * Draw the newly added point.
 * @return {void}
 */
function drawNew() {
    var i = clickX.length - 1
    if (!clickDrag[i]) {
        if (clickX.length == 0) {
            context.beginPath();
            context.moveTo(clickX[i], clickY[i]);
            context.stroke();
        } else {
            context.closePath();

            context.beginPath();
            context.moveTo(clickX[i], clickY[i]);
            context.stroke();
        }
    } else {
        context.lineTo(clickX[i], clickY[i]);
        context.stroke();
    }
}

function mouseDownEventHandler(e) {
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    paint = true;
    var x = e.pageX - canvas.offsetLeft;
    var y = e.pageY - canvas.offsetTop;
    if (paint) {
        addClick(x, y, false);
        drawNew();
    }
}

function touchstartEventHandler(e) {
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    paint = true;
    if (paint) {
        addClick(e.touches[0].pageX - canvas.offsetLeft, e.touches[0].pageY - canvas.offsetTop, false);
        drawNew();
    }
}

function mouseUpEventHandler(e) {
    context.closePath();
    paint = false;
    infer();
}

function mouseMoveEventHandler(e) {
    var x = e.pageX - canvas.offsetLeft;
    var y = e.pageY - canvas.offsetTop;
    if (paint) {
        addClick(x, y, true);
        drawNew();
    }
}

function touchMoveEventHandler(e) {
    if (paint) {
        addClick(e.touches[0].pageX - canvas.offsetLeft, e.touches[0].pageY - canvas.offsetTop, true);
        drawNew();
    }
}

function setUpHandler(isMouseandNotTouch, detectEvent) {
    removeRaceHandlers();
    if (isMouseandNotTouch) {
        canvas.addEventListener('mouseup', mouseUpEventHandler);
        canvas.addEventListener('mousemove', mouseMoveEventHandler);
        canvas.addEventListener('mousedown', mouseDownEventHandler);
        mouseDownEventHandler(detectEvent);
    } else {
        canvas.addEventListener('touchstart', touchstartEventHandler);
        canvas.addEventListener('touchmove', touchMoveEventHandler);
        canvas.addEventListener('touchend', mouseUpEventHandler);
        touchstartEventHandler(detectEvent);
    }
}

function mouseWins(e) {
    setUpHandler(true, e);
}

function touchWins(e) {
    setUpHandler(false, e);
}

function removeRaceHandlers() {
    canvas.removeEventListener('mousedown', mouseWins);
    canvas.removeEventListener('touchstart', touchWins);
}

let model = null;
let model_loaded = false;

let load_model = async () => {
    model = await tf.loadGraphModel("static/model/model.json");
    model_loaded = true;
};
load_model();

let predict = async (context, w, h) => {
    let img = context.getImageData(0, 0, w, h);
    let x = tf.browser.fromPixels(img).resizeBilinear([28, 28]).mean(2).reshape([1, 1, 28, 28]);
    let y = await model.predict({"img:0": x}, "Identity:0");
    let label = await tf.argMax(y.squeeze(0), 0).dataSync()[0];
    return label;
};

let infer = async () => {
    if(model_loaded) {
        let label = await predict(context, canvas.width, canvas.height);
        console.log(label);
        for(let i = 0; i < 10; i++) {
            if (label == i) {
                console.log(i);
                digits[i].content.bar.classList.add("argmax");
                digits[i].content.label.classList.add("argmax");
            } else {
                digits[i].content.bar.classList.remove("argmax");
                digits[i].content.label.classList.remove("argmax");
            }
        }
    }
}

canvas.addEventListener('mousedown', mouseWins);
canvas.addEventListener('touchstart', touchWins);