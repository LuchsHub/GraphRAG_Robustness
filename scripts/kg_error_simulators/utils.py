import csv


def get_deduplicated_triples_from_csv(input_file: str) -> tuple[list, set]:
    dedup_triples = set()

    with open(input_file, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)

        for row in reader:
            src_id = row[0]
            rel = row[1]
            dst_id = row[2]

            sorted_triple = (
                (src_id, rel, dst_id) if src_id < dst_id else (dst_id, rel, src_id)
            )

            if sorted_triple not in dedup_triples:
                dedup_triples.add(sorted_triple)

    return header, dedup_triples
