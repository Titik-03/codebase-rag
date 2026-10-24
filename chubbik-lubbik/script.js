// Small helper to select an element from the page.
const $ = (selector) => document.querySelector(selector);

function show(showChat) {
    $('#hero').hidden = showChat;
    $('#bottom').hidden = showChat;
    $('#chat').hidden = !showChat;

    if (showChat) {
        $('#q').focus();
    }
}

$('#start').onclick = () => show(true);
$('#navstart').onclick = () => show(true);
$('#home').onclick = () => show(false);

let text = 'RAG means retrieval augmented generation. It retrieves relevant passages from your documents before generating a grounded answer. Chubbik lubbik is your knowledge, a little magic. This interface is a browser-local demo; connect your local RAG backend for AI answers.';

$('#add').onclick = () => $('#file').click();

$('#file').onchange = async (event) => {
    const file = event.target.files[0];
    if (file) {
        text = await file.text();
        $('#name').textContent = file.name;
    }
};

$('form').onsubmit = (event) => {
    event.preventDefault();

    const terms = $('#q').value
        .toLowerCase()
        .split(/\W+/)
        .filter((word) => word.length > 2);

    const hits = text
        .split(/\n+|(?<=[.!?])\s+/)
        .map((passage) => ({
            t: passage,
            n: terms.filter((word) => passage.toLowerCase().includes(word)).length
        }))
        .filter((result) => result.n)
        .sort((a, b) => b.n - a.n)
        .slice(0, 3);

    $('#answer').hidden = false;
    $('#answer').textContent = hits.length
        ? 'Matching passages\n\n' + hits.map((result) => result.t).join('\n\n')
        : 'No matching passages. Try another question or add a text file.';
};
