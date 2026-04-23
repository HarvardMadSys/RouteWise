"""Tree Cache for Hybrid Inference Routing.

This module implements a rough approximation of a cache tree structure.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

from .const import MAX_TREE_CACHE_SIZE, MAX_TREE_NODE_LENGTH


@dataclass
class TreeNode:
    """A node in the tree structure. The maximum text length is MAX_TREE_NODE_LENGTH."""

    text: str
    children: dict[str, TreeNode]
    tenant_last_accessed_time: float
    parent: TreeNode | None


class TreeCache:
    """A cache for storing tree nodes and their associated information.

    All tree nodes are split into pieces with length <= MAX_TREE_NODE_LENGTH.
    Root node has empty text.
    All non-leaf nodes have exactly text length == MAX_TREE_NODE_LENGTH.
    Only leaf nodes can have text length < MAX_TREE_NODE_LENGTH.
    Do eviction base on LRU.

    Interfaces:
    - insert(text: str): Insert a new node with the given text into the tree.
    - match_max_prefix(text: str, update_access_time: bool) -> int: Match the longest prefix of the given text in the tree.
    - evict(): Evict the least recently used node from the tree if the cache size exceeds MAX_TREE_CACHE_SIZE.
    """

    def __init__(self, max_size: int = 1000):
        self.root = TreeNode(
            text="", children={}, tenant_last_accessed_time=time.time(), parent=None
        )
        self.text_count = 0

    def _create_new_node(
        self, text: str, parent: TreeNode, time_now: float = time.time()
    ) -> TreeNode:
        """Create a new tree node with the given text and parent."""
        self.text_count += len(text)
        new_node = TreeNode(
            text=text, children={}, tenant_last_accessed_time=time_now, parent=parent
        )
        parent.children[text] = new_node
        return new_node

    def _delete_leaf_node(self, node: TreeNode):
        """Delete a leaf node from the tree."""
        if node.children:
            raise ValueError("Cannot delete a non-leaf node.")
        if node.parent:
            del node.parent.children[node.text]
        self.text_count -= len(node.text)

    def _is_root(self, node: TreeNode) -> bool:
        return node.parent is None

    def insert(self, text: str):
        """Insert a new node with the given text into the tree.

        This method will keep matching the nodes in the tree until it finds a mismatch or reaches a leaf node.
        Note that the tree nodes can store more than one characters.
        """
        current_node = self.root
        current_index = 0
        text_length = len(text)
        time_now = time.time()
        while current_index < text_length:
            segment = text[current_index : current_index + MAX_TREE_NODE_LENGTH]
            if segment in current_node.children:
                # Hit existing node
                current_node = current_node.children[segment]
                current_index += len(segment)
                current_node.tenant_last_accessed_time = time_now
            else:
                # Create new node
                current_node = self._create_new_node(
                    text=segment, parent=current_node, time_now=time_now
                )
                current_index += len(segment)

    def match_max_prefix(self, text: str, update_access_time: bool) -> int:
        """Match the longest prefix of the given text in the tree.

        Returns the matched prefix.
        """
        current_node = self.root
        current_index = 0
        text_length = len(text)
        time_now = time.time() if update_access_time else None
        while current_index < text_length:
            segment = text[current_index : current_index + MAX_TREE_NODE_LENGTH]
            if segment in current_node.children:
                # Hit existing node
                current_node = current_node.children[segment]
                current_index += len(segment)
                if update_access_time:
                    current_node.tenant_last_accessed_time = time_now
            else:
                break
        return current_index

    def evict(self):
        """Evict the least recently used node from the tree."""
        if self.text_count <= MAX_TREE_CACHE_SIZE:
            return
        leaves = self._collect_leaves()
        heapq.heapify(leaves, key=lambda node: node.tenant_last_accessed_time)
        while self.text_count > MAX_TREE_CACHE_SIZE and leaves:
            lru_node = heapq.heappop(leaves)  # Root is not in the leaves list
            parent = lru_node.parent
            self._delete_leaf_node(lru_node)
            if parent and len(parent.children) == 1 and not self._is_root(parent):
                heapq.heappush(leaves, parent)

    def _collect_leaves(self) -> list[TreeNode]:
        """Collect all leaf nodes in the tree except the root.

        Returns a list of leaf nodes.
        """
        leaves = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not node.children and not self._is_root(node):
                leaves.append(node)
            for child in node.children.values():
                stack.append(child)
        return leaves
