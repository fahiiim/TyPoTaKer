from typotaker.infer import correct_sentence, load_run, next_words


def show(label: str, text: str) -> None:
    print(label, text.encode("unicode_escape").decode("ascii"))


def main() -> None:
    gec, tok = load_run("checkpoints/gec_stage0", "data/tokenizer/tokenizer.json")
    src = "Its important to chekc their config before you deploy."
    show("IN ", src)
    show("OUT", correct_sentence(gec, tok, src, "data/processed/gec/edit_maps.json"))
    lm, tok2 = load_run("checkpoints/lm_stage0", "data/tokenizer/tokenizer.json")
    print("NEXT")
    for piece, p in next_words(lm, tok2, "The server listens"):
        show(f" {p:.3f}", piece)


if __name__ == "__main__":
    main()
