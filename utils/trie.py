# -*- coding: utf-8 -*-

from collections import defaultdict


class TrieNode(object):
    def __init__(self):
        self.child = defaultdict(TrieNode)
        self.end = False


class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for c in word:
            node = node.child[c]
        node.end = True

    def search(self, word):
        node = self.root
        for c in word:
            if c not in node.child:
                return False
            else:
                node = node.child[c]
        return node.end

    def startswith(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node.child:
                return False
            else:
                node = node.child[c]
        return True

    def match(self, text):
        results = []
        for start in range(len(text)):
            node = self.root
            for length, c in enumerate(text[start:], start=1):
                if c not in node.child:
                    break
                node = node.child[c]
                if node.end:
                    results.append((start, start+length-1))
        return results
