from collections import namedtuple


class Span:
    def __init__(self, n_best_size, max_span_length, top_K):
        self.n_best_size = n_best_size
        self.max_span_length = max_span_length
        self.top_K = top_K

    def make(self, entities, max_seq_length):
        """
        entities: [(3,5), (5,6)]
        => [(3,4), (5,5)]
        """
        start_positions = [0 for _ in range(max_seq_length)]
        end_positions = [0 for _ in range(max_seq_length)]
        for s, e in entities:
            start_positions[s] = 1
            end_positions[e-1] = 1
        return start_positions, end_positions

    def parse(self, start_logits, end_logits, logit_threshold=1.,
              start_logit_t=0., end_logit_t=0., pn=-1, overlap_loc=[]):

        top_K = self.top_K if pn==-1 else pn
        candidates = span_annotate_candidates(
            start_logits, end_logits, self.n_best_size,
            self.max_span_length, logit_threshold,
            top_K, start_logit_t, end_logit_t, overlap_loc)

        return [(s, e+1) for s,e in candidates]


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def span_annotate_candidates(start_logits, end_logits, n_best_size,
                             max_span_length, logit_threshold, top_K,
                             start_logit_t, end_logit_t, overlap_loc):

    _PrelimPrediction = namedtuple(
        "_PrelimPrediction",
        ['start_index', 'end_index', 'start_logit', 'end_logit']
    )
    candidates = []
    prelim_predictions_per_feature = []

    start_indexes = _get_best_indexes(start_logits, n_best_size)
    end_indexes = _get_best_indexes(end_logits, n_best_size)

    for start_index in start_indexes:
        for end_index in end_indexes:
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_span_length:
                continue
            start_logit = start_logits[start_index]
            end_logit = end_logits[end_index]
            if start_logit + end_logit < logit_threshold:
                continue
            # 为了真实值
            if start_logit <= start_logit_t or end_logit <= end_logit_t:
                continue
            # if (start_index, end_index) in overlap_loc:
            #     print(start_index, end_index, overlap_loc)
            #     continue

            continue_flag = False
            for s, e in overlap_loc:
                if start_index<=s and e<=end_index:
                    continue_flag = True
                    break
            if continue_flag:
                continue

            prelim_predictions_per_feature.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logit,
                    end_logit=end_logit,
                )
            )

    prelim_predictions_per_feature = sorted(
        prelim_predictions_per_feature,
        key=lambda x: (x.start_logit + x.end_logit - (x.end_index - x.start_index + 1)),
        reverse=True,
    )

    for i, pred_i in enumerate(prelim_predictions_per_feature):
        if len(candidates) >= top_K:
            break

        candidates.append((pred_i.start_index, pred_i.end_index))
        filter_redundant_candidates(prelim_predictions_per_feature, i, candidates[-1])

    return candidates


def filter_redundant_candidates(prelim_predictions_per_feature, i, candidate):
    start_index, end_index = candidate
    indexes = []
    for j, pred_j in enumerate(prelim_predictions_per_feature[i+1:]):
        if overlap(start_index, end_index, pred_j.start_index, pred_j.end_index):
            indexes.append(i+j+1)

    [prelim_predictions_per_feature.pop(index-k) for k, index in enumerate(indexes)]


def overlap(s1, e1, s2, e2):
    # [s1, e1+1] [s2, e2+1]
    # [s2, e2+1] [s1, e1+1]
    if s1 <= e1 < s2 <= e2:
        return False
    if s2 <= e2 < s1 <= e1:
        return False
    return True


if __name__ == '__main__':
    span = Span(20, 5, 1.01, 12, 10)
    s, e = span.make([(3, 5), (5, 6)])
    print(s)
    print(e)
    cans = span.parse(s, e)
    print(cans)
