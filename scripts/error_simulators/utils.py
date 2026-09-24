import csv


def get_deduplicated_triples_from_csv(input_file: str) -> tuple[list, set, int]:
    dedup_triples = set()

    max_id = -1
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

            if sorted_triple in dedup_triples:
                continue

            if int(src_id) > max_id:
                max_id = int(src_id)
            if int(dst_id) > max_id:
                max_id = int(dst_id)

            dedup_triples.add(sorted_triple)

    return header, dedup_triples, max_id
