// $.ajax({
//     type: "GET",
//     url: "/get-predicted-sentiment/",
//     success: function (response) {
//         console.log(">> Success: ", response)
//     },
//     error: function (error) {
//         console.error(">> Error: ", error)
//     }
// });

const commentArea = document.getElementById('commentArea')
const negaProb = document.getElementById('negaProb')
const posiProb = document.getElementById('posiProb')
const mainForm = document.getElementById('mainForm')
const getCookie = (name) => {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
const csrftoken = getCookie('csrftoken');

mainForm.addEventListener('submit', e => {
    e.preventDefault()

    let text = commentArea.value.trim()
    if (text !== "") {
        $.ajax({
            type: "POST",
            url: "/ajax-predict-comment/",
            data: {
                'csrfmiddlewaretoken': csrftoken,
                'comment': commentArea.value
            },
            success: function (response) {
                console.log(response)
                negaProb.textContent = "negative: " + (response.negaProb * 100.0).toFixed(2) + "%"
                posiProb.textContent = "positive: " + (response.posiProb * 100.0).toFixed(2) + "%"
            },
            error: function(error) {
                console.log(error)
            }
        });
    } else {
        alert("Your text is empty!")
    }
});