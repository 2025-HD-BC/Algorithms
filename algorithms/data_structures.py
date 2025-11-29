"""
Data Structures Implementation

Clean implementations of fundamental data structures with proper documentation
and complexity analysis.

Author: Gert Coetser
Date: November 2025
"""

from typing import Any, Optional, List
from dataclasses import dataclass


# ============================================================================
# LINKED LIST
# ============================================================================

@dataclass
class ListNode:
    """Node for singly linked list."""
    val: Any
    next: Optional['ListNode'] = None


class LinkedList:
    """
    Singly Linked List implementation.
    
    Time Complexity:
        - Insert at head: O(1)
        - Insert at tail: O(1) with tail pointer
        - Search: O(n)
        - Delete: O(n)
    """
    
    def __init__(self):
        """Initialize empty linked list."""
        self.head: Optional[ListNode] = None
        self.tail: Optional[ListNode] = None
        self.size: int = 0
    
    def append(self, val: Any) -> None:
        """
        Add value to the end of the list.
        
        Time Complexity: O(1)
        Args:
            val: Value to append
        """
        new_node = ListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def prepend(self, val: Any) -> None:
        """
        Add value to the beginning of the list.
        
        Time Complexity: O(1)
        Args:
            val: Value to prepend
        """
        new_node = ListNode(val, self.head)
        self.head = new_node
        
        if not self.tail:
            self.tail = new_node
        
        self.size += 1
    
    def find(self, val: Any) -> bool:
        """
        Check if value exists in the list.
        
        Time Complexity: O(n)
        Args:
            val: Value to find
            
        Returns:
            True if found, False otherwise
        """
        current = self.head
        
        while current:
            if current.val == val:
                return True
            current = current.next
        
        return False
    
    def delete(self, val: Any) -> bool:
        """
        Delete first occurrence of value.
        
        Time Complexity: O(n)
        Args:
            val: Value to delete
            
        Returns:
            True if deleted, False if not found
        """
        if not self.head:
            return False
        
        # Special case: delete head
        if self.head.val == val:
            self.head = self.head.next
            if not self.head:
                self.tail = None
            self.size -= 1
            return True
        
        # Find and delete
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                if not current.next:
                    self.tail = current
                self.size -= 1
                return True
            current = current.next
        
        return False
    
    def to_list(self) -> List[Any]:
        """Convert linked list to Python list."""
        result = []
        current = self.head
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    def __len__(self) -> int:
        """Return size of the list."""
        return self.size
    
    def __str__(self) -> str:
        """String representation of the list."""
        return " -> ".join(str(val) for val in self.to_list())


# ============================================================================
# BINARY SEARCH TREE
# ============================================================================

@dataclass
class TreeNode:
    """Node for binary search tree."""
    val: int
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None


class BinarySearchTree:
    """
    Binary Search Tree implementation.
    
    Time Complexity (average case):
        - Insert: O(log n)
        - Search: O(log n)
        - Delete: O(log n)
        
    Time Complexity (worst case - unbalanced):
        - All operations: O(n)
    """
    
    def __init__(self):
        """Initialize empty BST."""
        self.root: Optional[TreeNode] = None
    
    def insert(self, val: int) -> None:
        """
        Insert a value into the BST.
        
        Args:
            val: Value to insert
        """
        if not self.root:
            self.root = TreeNode(val)
        else:
            self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node: TreeNode, val: int) -> None:
        """Helper method for recursive insertion."""
        if val < node.val:
            if node.left is None:
                node.left = TreeNode(val)
            else:
                self._insert_recursive(node.left, val)
        else:
            if node.right is None:
                node.right = TreeNode(val)
            else:
                self._insert_recursive(node.right, val)
    
    def search(self, val: int) -> bool:
        """
        Search for a value in the BST.
        
        Args:
            val: Value to search for
            
        Returns:
            True if found, False otherwise
        """
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node: Optional[TreeNode], val: int) -> bool:
        """Helper method for recursive search."""
        if node is None:
            return False
        
        if val == node.val:
            return True
        elif val < node.val:
            return self._search_recursive(node.left, val)
        else:
            return self._search_recursive(node.right, val)
    
    def inorder_traversal(self) -> List[int]:
        """
        Perform inorder traversal (left, root, right).
        Returns sorted list for BST.
        
        Returns:
            List of values in inorder
        """
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node: Optional[TreeNode], result: List[int]) -> None:
        """Helper method for inorder traversal."""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.val)
            self._inorder_recursive(node.right, result)
    
    def find_min(self) -> Optional[int]:
        """Find minimum value in the BST."""
        if not self.root:
            return None
        
        current = self.root
        while current.left:
            current = current.left
        return current.val
    
    def find_max(self) -> Optional[int]:
        """Find maximum value in the BST."""
        if not self.root:
            return None
        
        current = self.root
        while current.right:
            current = current.right
        return current.val


# ============================================================================
# MIN HEAP
# ============================================================================

class MinHeap:
    """
    Binary Min Heap implementation.
    
    Time Complexity:
        - Insert: O(log n)
        - Extract Min: O(log n)
        - Get Min: O(1)
        - Heapify: O(n)
    """
    
    def __init__(self):
        """Initialize empty heap."""
        self.heap: List[int] = []
    
    def parent(self, i: int) -> int:
        """Get parent index."""
        return (i - 1) // 2
    
    def left_child(self, i: int) -> int:
        """Get left child index."""
        return 2 * i + 1
    
    def right_child(self, i: int) -> int:
        """Get right child index."""
        return 2 * i + 2
    
    def insert(self, val: int) -> None:
        """
        Insert a value into the heap.
        
        Args:
            val: Value to insert
        """
        self.heap.append(val)
        self._bubble_up(len(self.heap) - 1)
    
    def _bubble_up(self, i: int) -> None:
        """Move element up to maintain heap property."""
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            # Swap with parent
            self.heap[i], self.heap[self.parent(i)] = \
                self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)
    
    def extract_min(self) -> Optional[int]:
        """
        Remove and return the minimum element.
        
        Returns:
            Minimum value or None if heap is empty
        """
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        # Store min value
        min_val = self.heap[0]
        
        # Move last element to root and bubble down
        self.heap[0] = self.heap.pop()
        self._bubble_down(0)
        
        return min_val
    
    def _bubble_down(self, i: int) -> None:
        """Move element down to maintain heap property."""
        min_index = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        # Find smallest among node and its children
        if left < len(self.heap) and self.heap[left] < self.heap[min_index]:
            min_index = left
        
        if right < len(self.heap) and self.heap[right] < self.heap[min_index]:
            min_index = right
        
        # If smallest is not the node itself, swap and continue
        if min_index != i:
            self.heap[i], self.heap[min_index] = \
                self.heap[min_index], self.heap[i]
            self._bubble_down(min_index)
    
    def peek(self) -> Optional[int]:
        """
        Get minimum element without removing it.
        
        Returns:
            Minimum value or None if heap is empty
        """
        return self.heap[0] if self.heap else None
    
    def size(self) -> int:
        """Return size of the heap."""
        return len(self.heap)
    
    def __str__(self) -> str:
        """String representation of the heap."""
        return str(self.heap)


# ============================================================================
# GRAPH
# ============================================================================

class Graph:
    """
    Graph implementation using adjacency list.
    
    Time Complexity:
        - Add vertex: O(1)
        - Add edge: O(1)
        - BFS: O(V + E)
        - DFS: O(V + E)
    """
    
    def __init__(self, directed: bool = False):
        """
        Initialize empty graph.
        
        Args:
            directed: True for directed graph, False for undirected
        """
        self.graph: dict[Any, List[Any]] = {}
        self.directed = directed
    
    def add_vertex(self, vertex: Any) -> None:
        """Add a vertex to the graph."""
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    def add_edge(self, v1: Any, v2: Any) -> None:
        """
        Add an edge between two vertices.
        
        Args:
            v1: First vertex
            v2: Second vertex
        """
        # Ensure both vertices exist
        self.add_vertex(v1)
        self.add_vertex(v2)
        
        # Add edge
        self.graph[v1].append(v2)
        
        # For undirected graph, add reverse edge
        if not self.directed:
            self.graph[v2].append(v1)
    
    def bfs(self, start: Any) -> List[Any]:
        """
        Breadth-First Search traversal.
        
        Args:
            start: Starting vertex
            
        Returns:
            List of vertices in BFS order
        """
        if start not in self.graph:
            return []
        
        visited = set()
        queue = [start]
        result = []
        
        while queue:
            vertex = queue.pop(0)
            
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                # Add unvisited neighbors to queue
                for neighbor in self.graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result
    
    def dfs(self, start: Any) -> List[Any]:
        """
        Depth-First Search traversal.
        
        Args:
            start: Starting vertex
            
        Returns:
            List of vertices in DFS order
        """
        if start not in self.graph:
            return []
        
        visited = set()
        result = []
        
        def dfs_recursive(vertex):
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    dfs_recursive(neighbor)
        
        dfs_recursive(start)
        return result
    
    def __str__(self) -> str:
        """String representation of the graph."""
        lines = []
        for vertex, neighbors in self.graph.items():
            lines.append(f"{vertex} -> {neighbors}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    print("Data Structures Demo\n" + "="*50)
    
    # LinkedList demo
    print("\n1. LinkedList:")
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    print(f"   List: {ll}")
    print(f"   Find 2: {ll.find(2)}")
    print(f"   Size: {len(ll)}")
    
    # BST demo
    print("\n2. Binary Search Tree:")
    bst = BinarySearchTree()
    for val in [5, 3, 7, 1, 9, 4]:
        bst.insert(val)
    print(f"   Inorder: {bst.inorder_traversal()}")
    print(f"   Search 7: {bst.search(7)}")
    print(f"   Min: {bst.find_min()}, Max: {bst.find_max()}")
    
    # MinHeap demo
    print("\n3. Min Heap:")
    heap = MinHeap()
    for val in [5, 3, 7, 1, 9, 4]:
        heap.insert(val)
    print(f"   Heap: {heap}")
    print(f"   Extract min: {heap.extract_min()}")
    print(f"   Peek: {heap.peek()}")
    
    # Graph demo
    print("\n4. Graph (BFS/DFS):")
    g = Graph()
    g.add_edge('A', 'B')
    g.add_edge('A', 'C')
    g.add_edge('B', 'D')
    g.add_edge('C', 'D')
    print(f"   BFS from A: {g.bfs('A')}")
    print(f"   DFS from A: {g.dfs('A')}")
    
    print("\n" + "="*50)
