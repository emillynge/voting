from pprintpp import pprint
from collections import (UserList, Counter, OrderedDict)
from random import shuffle, randint
from typing import List
from itertools import combinations

def vote_generator(n_cand, n_votes):
    """
    method for testing. returns synthetic votes in the form of shuffled lists of integers
    :param n_cand: Number of possible candidates in election
    :param n_votes: Desired number of votes
    :return: generator function
    """
    for _ in range(n_votes):
        prios = list(range(n_cand))
        shuffle(prios)
        yield prios[:randint(0, n_cand-1)]

def test_vote1():
    votes = [
        [1, 2, 3],
        [2, 3, 4],
        [2, 1],
        [4, 2, 1],
        [3, 1],
        [3, 4],
        [1, 4],
        [],
    ]
    round1 = [1, 2, 2, 4, 3, 3, 1, None] # remove 4
    round2 = [1, 2, 2, 2, 3, 3, 1, None] # remove 3
    round3 = [1, 2, 2, 2, 1, None, 1, None] # remove 2
    round3 = [1, None, 1, 1, 1, None, 1, None] # winner

    election = Election([1,2,3,4, 5], votes)
    election.find_winner()


def test_comb_votes():
    votes = [
        ['ab', 'bc'],
        ['de', 'ae'],
        ['ec', 'ab'],
    ]
    candidates = ['foo', 'bar', 'alice', 'bob', 'john']
    Election(candidates, votes, 'alphaind_comb').find_winner()


class Vote(UserList):
    def __init__(self, priorities):
        super().__init__(priorities)
        self.discarded = list()  # List of votes that have been discarded

    def discard(self, i):
        try:
            self.remove(i)
        except ValueError:
            pass
        self.discarded.append(i)

    def peek(self, i):
        """
        Peek at priority i. If no such exist return None (abstain).
        :return:
        """
        try:
            return self[i]
        except IndexError:
            return -1


class Votebox(UserList):
    def __init__(self, votes: List[Vote], n_candidates):
        super().__init__(votes)
        self.n_cand = n_candidates
        self.discarded = set()

    def discard_candidate(self, i):
        self.discarded.add(i)
        for vote in self:
            vote.discard(i)

    def count_in_tier(self, tier):
        """
        Count the votes in a tier.
        tier 0 is the first priorities in the votebox, tier 1 is second etc.
        :param tier:
        :return:
        """
        # initialize all non-discarded candidates to zero votes
        counter = Counter(dict((i, 0) for i in range(self.n_cand)
                               if i not in self.discarded))
        counter.update(vote.peek(tier) for vote in self)
        return counter


class InfeasibleElection(Exception): pass


class Election:
    def __init__(self, candidates: list, votes, vote_type='lookup'):
        self.candidates = candidates
        self.stats = dict()
        self.winner = None

        normalizer = dict(
            lookup=self.normalize_lookup_votes,
            none=lambda vote: vote,
            alphaind_comb=self.normalize_comb_alphaindex,
                          )[vote_type]

        self.votes = Votebox(normalizer(votes), len(self.candidates))
        self.set_stats()

    def normalize_comb_alphaindex(self, votes):
        alpha = 'abcdefghijklmn'
        cand = self.candidates
        self.candidates = list(' & '.join(sorted(comb))
                               for comb in combinations(cand, 2))
        def get_lookup_votes():
            for vote in votes:
                vote = [' & '.join(sorted(cand[alpha.index(al_in)] for al_in in prio))
                         for prio in vote]
                yield vote

        return self.normalize_lookup_votes(get_lookup_votes())









    def normalize_lookup_votes(self, votes):
        def lookup(vote, i):
            for prio in vote:
                try:
                    yield self.candidates.index(prio)
                except ValueError:
                    raise ValueError('vote contains priority "{}" not present'
                                     ' in candidates list: \n {}'
                                     ''.format(prio, self.candidates, i))

        for i, vote in enumerate(votes):
            yield Vote(lookup(vote, i))

    def find_loosers(self, tier, loser_cand: list):
        vote_count = self.votes.count_in_tier(tier)
        vote_count.pop(-1, 0)  # we don't need abstain votes
        if not any(vote_count.values()):
            raise InfeasibleElection('Cannot find a looser. No votes after reaching tier {}'
                                     ''.format(tier))


        least_common = list(vote_count.most_common())[::-1]
        least_votes = len(self.votes)  # Ensure higher than any vote count
        candidates = list()

        for cand, votes in least_common:
            if least_votes < votes:
                break
            if cand in loser_cand:
                least_votes = votes
                candidates.append(cand)

        if len(candidates) > 1:
            try:
                return self.find_loosers(tier + 1, loser_cand=candidates)
            except InfeasibleElection:
                if candidates != loser_cand:
                    return candidates
                raise
        return candidates

    def set_stats(self):
        count = self.votes.count_in_tier(0)
        count_orig = Counter(count)

        abstain_votes = count.pop(-1, 0)
        leader = count.most_common(1)[0][0]
        vote_percentage = 100 * (count[leader] / sum(count.values()))
        vote_percentage_abs = 100 * (count[leader] / len(self.votes))
        vote_percentage_abstain = 100 * (abstain_votes / len(self.votes))

        if vote_percentage > 50:
            self.winner = self.candidates[leader]

        self.stats.update({'percentages': {'leader_lead': vote_percentage,
                                      'leader_support': vote_percentage_abs,
                                      'abstain': vote_percentage_abstain},
                      'counts': OrderedDict(self.translate_count(count_orig)),
                      'leader': self.candidates[leader],
                      })

    def do_a_round(self):
        """
        Do a round of pruning and return election state AFTER pruning
        :return:
        """
        losers = self.find_loosers(0, [i for i in range(len(self.candidates))
                                       if i not in self.votes.discarded])
        for loser in losers:
            self.votes.discard_candidate(loser)
        self.stats['pruned'] = [self.candidates[i] for i in losers]
        self.set_stats()


    def find_winner(self):
        pprint(self.stats)
        while self.winner is None:
            self.do_a_round()
            pprint(self.stats)

        print('Winner: {}'.format(self.winner))

    def translate_count(self, count: Counter):
        for cand_i, votes in count.most_common():
            if cand_i == -1:
                yield 'abstain', votes
            else:
                yield self.candidates[cand_i], votes


test_comb_votes()