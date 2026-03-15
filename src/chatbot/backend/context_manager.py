from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field
from jinja2 import Environment, BaseLoader, select_autoescape
import logging

logger = logging.getLogger("llaisys.context_manager")

@dataclass
class TaggedItem:
    """
    Represents a context item with associated tags.
    """
    id: str
    content: str
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_tag(self, tag: str):
        self.tags.add(tag)

    def remove_tag(self, tag: str):
        self.tags.discard(tag)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags
    
    def has_any_tag(self, tags: List[str]) -> bool:
        return bool(self.tags.intersection(tags))
    
    def has_all_tags(self, tags: List[str]) -> bool:
        return self.tags.issuperset(tags)

class ContextManager:
    """
    Manages a collection of tagged context items and provides template rendering capabilities.
    """
    def __init__(self):
        self.items: Dict[str, TaggedItem] = {}
        
        # Initialize Jinja2 Environment
        self.env = Environment(
            loader=BaseLoader(), # Use string templates
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Register custom filters
        self.env.filters['filter_by_tag'] = self.filter_by_tag
        self.env.filters['exclude_by_tag'] = self.exclude_by_tag

    def add_item(self, item_id: str, content: str, tags: List[str] = None, metadata: Dict[str, Any] = None) -> TaggedItem:
        """Adds or updates a context item."""
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}
            
        item = TaggedItem(
            id=item_id,
            content=content,
            tags=set(tags),
            metadata=metadata
        )
        self.items[item_id] = item
        logger.info(f"Added context item: {item_id} with tags: {tags}")
        return item

    def get_item(self, item_id: str) -> Optional[TaggedItem]:
        return self.items.get(item_id)

    def remove_item(self, item_id: str) -> bool:
        if item_id in self.items:
            del self.items[item_id]
            logger.info(f"Removed context item: {item_id}")
            return True
        return False

    def list_items(self) -> List[TaggedItem]:
        return list(self.items.values())

    def get_items_by_tag(self, tag: str) -> List[TaggedItem]:
        """Returns all items that have the specified tag."""
        return [item for item in self.items.values() if item.has_tag(tag)]

    def filter_by_tag(self, items: List[TaggedItem], tag: str) -> List[TaggedItem]:
        """Jinja2 filter: Select items that have the specified tag."""
        if not items:
            return []
        filtered = []
        for item in items:
            if isinstance(item, TaggedItem) and item.has_tag(tag):
                filtered.append(item)
            elif isinstance(item, dict) and tag in item.get('tags', []):
                filtered.append(item)
        return filtered

    def exclude_by_tag(self, items: List[TaggedItem], tag: str) -> List[TaggedItem]:
        """Jinja2 filter: Exclude items that have the specified tag."""
        if not items:
            return []
        filtered = []
        for item in items:
            if isinstance(item, TaggedItem) and not item.has_tag(tag):
                filtered.append(item)
        return filtered

    def render_template(self, template_str: str, extra_context: Dict[str, Any] = None) -> str:
        """
        Renders a template string using the current context items and optional extra context.
        The template has access to 'context_items' which is the list of all TaggedItems.
        """
        try:
            template = self.env.from_string(template_str)
            render_context = {
                "context_items": list(self.items.values()),
                **(extra_context or {})
            }
            return template.render(**render_context)
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            raise

if __name__ == "__main__":
    cm = ContextManager()
    cm.add_item("rule1", "Always be polite.", ["behavior", "core"])
    cm.add_item("rule2", "Use concise language.", ["style", "core"])
    cm.add_item("fact1", "The sky is blue.", ["knowledge", "nature"])
    
    template = """
    System Instructions:
    {% for item in context_items | filter_by_tag('core') %}
    - {{ item.content }}
    {% endfor %}
    
    Knowledge:
    {% for item in context_items | filter_by_tag('knowledge') %}
    - {{ item.content }}
    {% endfor %}
    """
    
    print(cm.render_template(template))
