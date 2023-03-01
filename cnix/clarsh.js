let user_input = "";
let shift_pressed = false;
let term_element = null;


function sh_prompt(path) {
    return "<font color='#4E9A06'>user@cnix</font>:<font color='#3465A4'>" + path + "</font>$ ";
}


document.onkeydown = function(e) {
    e = e || window.event;
    console.log(e.keyCode);
    let key = e.keyCode;

    if (32 <= key && key <= 126) {
        // Printable characters
        if (!shift_pressed && 65 <= key && key <= 90) {
            // Lowercase
            key += 32;
        }
        user_input += String.fromCharCode(key);
    } else if (key == 8) {
        // Backspace
        user_input = user_input.slice(0, -1);
    } else if (key == 16) {
        // Shift
        shift_pressed = true;
    } else if (key == 13) {
        // Enter
        // Create new element in body
        update(false);

        let new_element = document.createElement("div");
        document.body.appendChild(new_element);
        term_element = new_element;

        // TODO execute command
        user_input = "";
    }
}

document.onkeyup = function(e) {
    e = e || window.event;
    const key = e.keyCode;

    if (key == 16) {
        // Shift
        shift_pressed = false;
    }
}


function update(do_blink = true) {
    if (term_element == null) {
        term_element = document.getElementById("terminal");
    }

    let prompt = sh_prompt("/home/user") + user_input;

    // Blinking cursor
    let time = (new Date()).getTime();
    let blink = (time % 1000) < 500;
    if (do_blink && blink) {
        prompt += "<font style='background-color:white;'>&nbsp;</font>";
    }

    term_element.innerHTML = prompt;
}


function initialize() {
    let t = setInterval(update, 100);
}
