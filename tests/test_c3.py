import tempfile
from pathlib import Path
from app.context import ContextManager
from app.search import Candidate
from app.memory.graph import IdentityKnowledgeGraph

def test_context_manager_extracts_tags():
    """Verify ContextManager extracts hashtags, handles, and proper nouns correctly."""
    cm = ContextManager()
    
    # Simulate candidates with rich OSINT text
    candidates = [
        Candidate(
            title="Spotted at Hacker House Goa 2026. @john_doe was there! #tech #AI",
            domain="github.com",
            source_url="http://github.com/post1",
            image_url="http://github.com/img1.jpg"
        ),
        Candidate(
            title="Stanford University researchers publish new findings. stanford.edu",
            domain="stanford.edu",
            source_url="http://stanford.edu/news",
            image_url="http://stanford.edu/img2.jpg"
        )
    ]
    
    raw_tags = cm.extract_tags(candidates)
    
    # Should contain lowercase hashtags
    assert "#tech" in raw_tags
    assert "#ai" in raw_tags
    
    # Should contain handles
    assert "@john_doe" in raw_tags
    
    # Should extract proper nouns (capitalized) but not stop words
    assert "hacker" in raw_tags
    assert "house" in raw_tags
    
    # Should extract domain bases
    assert "github" in raw_tags
    assert "stanford" in raw_tags

def test_context_manager_scoring_algorithm():
    """Verify ContextManager boosts handles and hashtags above standard nouns."""
    cm = ContextManager()
    
    # Deliberately unbalanced input: "apple" appears 3 times, "@john" appears 2 times.
    raw_tags = ["apple", "apple", "apple", "@john", "@john", "tech", "#ai"]
    
    ranked_tags = cm.score_and_rank_tags(raw_tags)
    
    # Score logic:
    # "apple" = 3 count -> 3.0
    # "@john" = 2 count * 2.0 (boost) = 4.0
    # So "@john" should outrank "apple" despite lower absolute frequency.
    
    assert ranked_tags[0] == "@john"
    assert ranked_tags[1] == "apple"

def test_identity_knowledge_graph_pending_targets():
    """Verify the graph can ingest and resolve pending targets in an isolated file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test_graph.json"
        
        # Instantiate with a temporary file so we don't pollute the actual knowledge graph
        graph = IdentityKnowledgeGraph(store_path=store_path)
        
        # Ensure it starts empty
        assert len(graph.get_pending_targets()) == 0
        
        # Ingest a new target
        dummy_embedding = [0.1, 0.2, 0.3, 0.4] * 128  # 512-d mock embedding
        person = graph.add_pending_target(embedding=dummy_embedding, image_path="http://test.com/face.jpg")
        
        # Verify it was added as pending
        pending = graph.get_pending_targets()
        assert len(pending) == 1
        assert pending[0].status == "pending"
        assert pending[0].id == person.id
        
        # Verify it resolves correctly
        graph.mark_resolved(person.id)
        
        # It should no longer be pending
        assert len(graph.get_pending_targets()) == 0
        
        # It should now be in the resolved targets list
        resolved = graph.get_resolved_targets()
        assert len(resolved) == 1
        assert resolved[0].status == "resolved"
        assert resolved[0].id == person.id
        
        # Verify persistence (re-loading the graph from the temp file)
        new_graph = IdentityKnowledgeGraph(store_path=store_path)
        assert len(new_graph.get_pending_targets()) == 0
        assert len(new_graph.get_resolved_targets()) == 1
        assert new_graph.get_resolved_targets()[0].id == person.id
