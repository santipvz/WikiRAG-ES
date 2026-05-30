from mitex_assistant.data.models import Document
from mitex_assistant.indexing.chunking import chunk_document


def _doc(text: str) -> Document:
    return Document(doc_id="d1", text=text, source_path="mem", title="t")


def test_chunking_covers_full_text_with_overlap() -> None:
    text = ". ".join(f"frase numero {i} con relleno" for i in range(200))
    chunks = chunk_document(_doc(text), chunk_size=300, overlap=50)

    assert len(chunks) > 1
    # Cada chunk respeta el tamano maximo (con margen por el corte en frontera).
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    # Hay solape: el inicio de un chunk precede al fin del anterior.
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start_char < previous.end_char


def test_chunking_short_text_returns_single_chunk() -> None:
    chunks = chunk_document(_doc("Un texto corto."), chunk_size=900, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == "Un texto corto."


def test_chunking_rejects_overlap_ge_chunk_size() -> None:
    try:
        chunk_document(_doc("hola mundo"), chunk_size=100, overlap=100)
    except ValueError:
        return
    raise AssertionError("Se esperaba ValueError cuando overlap >= chunk_size")
