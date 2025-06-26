import json
import os
import re
from typing import List, Tuple
from collections import Counter
import argparse
from dotenv import load_dotenv
import time

load_dotenv()

# OpenAI imports
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Install with: pip install openai")

# Groq imports
try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: Groq library not installed. Install with: pip install groq")

# spaCy imports
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy library not installed. Install with: pip install spacy")


class TagGenerator:
    def __init__(self, technique: str = "spacy"):
        """
        Initialize the tag generator with specified technique.

        Args:
            technique: One of "spacy", "openai", or "groq"
        """
        self.technique = technique.lower()

        if self.technique == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library is required for openai technique")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            self.openai_client = OpenAI(api_key=api_key)

        elif self.technique == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Groq library is required for groq technique")
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is required")
            self.groq_client = Groq(api_key=api_key)

        elif self.technique == "spacy":
            if not SPACY_AVAILABLE:
                raise ImportError("spaCy library is required for spacy technique")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy English model not found. Download with: python -m spacy download en_core_web_sm")
                raise
        else:
            raise ValueError("technique must be either 'openai', 'groq', or 'spacy'")

        # Always initialize spaCy for fallback
        if self.technique != "spacy" and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None

    def generate_tags_with_weights(self, text: str, max_tags: int = 10) -> List[Tuple[str, float]]:
        """
        Generate tags with importance weights for a given text.

        Args:
            text: Input text to analyze
            max_tags: Maximum number of tags to return

        Returns:
            List of (tag, weight) tuples sorted by weight descending
        """
        if self.technique == "openai":
            return self._generate_tags_openai(text, max_tags)
        elif self.technique == "groq":
            return self._generate_tags_groq(text, max_tags)
        else:
            return self._generate_tags_spacy(text, max_tags)

    def _generate_tags_openai(self, text: str, max_tags: int) -> List[Tuple[str, float]]:
        """Generate tags using OpenAI API"""
        prompt = f"""
        Analyze the following text and generate {max_tags} relevant conceptual tags with importance weights (0.0 to 1.0).
        Tags should be abstract themes and concepts (like "time management", "personal development"), not just keywords.
        Return ONLY a JSON array with "tag" and "weight" fields.

        Text: {text[:1000]}

        Format: [{{"tag": "example", "weight": 0.8}}]
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate conceptual tags as JSON. No explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            if not content.startswith('['):
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    content = match.group(0)

            tags_data = json.loads(content)

            # Process and validate tags
            tags_with_weights = []
            for item in tags_data:
                if isinstance(item, dict) and 'tag' in item and 'weight' in item:
                    tag = str(item['tag']).lower().strip()
                    weight = float(item['weight'])
                    if tag and 0.0 <= weight <= 1.0:
                        tags_with_weights.append((tag, weight))

            return sorted(tags_with_weights, key=lambda x: x[1], reverse=True)[:max_tags]

        except Exception as e:
            print(f"Error with OpenAI: {e}")
            return self._generate_tags_spacy(text, max_tags)

    def _generate_tags_groq(self, text: str, max_tags: int) -> List[Tuple[str, float]]:
        """Generate tags using Groq API"""
        prompt = f"""
        Analyze the following text and generate {max_tags} relevant conceptual tags with importance weights (0.0 to 1.0).
        Tags should be abstract themes and concepts (like "time management", "personal development"), not just keywords.
        Return ONLY a JSON array with "tag" and "weight" fields.

        Text: {text[:1000]}

        Format: [{{"tag": "example", "weight": 0.8}}]
        """

        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "Generate conceptual tags as JSON. No explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            if not content.startswith('['):
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    content = match.group(0)

            tags_data = json.loads(content)

            # Process and validate tags
            tags_with_weights = []
            for item in tags_data:
                if isinstance(item, dict) and 'tag' in item and 'weight' in item:
                    tag = str(item['tag']).lower().strip()
                    weight = float(item['weight'])
                    if tag and 0.0 <= weight <= 1.0:
                        tags_with_weights.append((tag, weight))

            return sorted(tags_with_weights, key=lambda x: x[1], reverse=True)[:max_tags]

        except Exception as e:
            print(f"Error with Groq: {e}")
            return self._generate_tags_spacy(text, max_tags)

    def _generate_tags_spacy(self, text: str, max_tags: int) -> List[Tuple[str, float]]:
        """Generate tags using spaCy NLP"""
        if not hasattr(self, 'nlp') or self.nlp is None:
            # If spaCy not loaded, return empty list
            print("spaCy not available for fallback")
            return []

        doc = self.nlp(text)
        tag_scores = {}

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT"]:
                clean_tag = self._clean_tag(ent.text)
                if clean_tag:
                    tag_scores[clean_tag] = tag_scores.get(clean_tag, 0) + 1.0

        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if 1 <= len(chunk.text.split()) <= 3:
                clean_tag = self._clean_tag(chunk.text)
                if clean_tag and clean_tag not in STOP_WORDS:
                    weight = 0.8 if len(chunk.text.split()) > 1 else 0.6
                    tag_scores[clean_tag] = tag_scores.get(clean_tag, 0) + weight

        # Extract important nouns
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) > 2:
                clean_tag = self._clean_tag(token.lemma_)
                if clean_tag:
                    tag_scores[clean_tag] = tag_scores.get(clean_tag, 0) + 0.5

        # Normalize scores
        if tag_scores:
            max_score = max(tag_scores.values())
            tag_scores = {tag: score / max_score for tag, score in tag_scores.items()}

        return sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)[:max_tags]

    def _clean_tag(self, tag: str) -> str:
        """Clean and normalize a tag"""
        # Remove special characters and normalize
        cleaned = re.sub(r'[^\w\s-]', '', tag.strip().lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Skip if too short or common
        if len(cleaned) < 2 or cleaned in STOP_WORDS:
            return ""

        return cleaned

    def process_articles_file(self, input_file: str, output_file: str, max_tags: int = 10):
        """
        Process articles from input JSON file and generate tags for each.

        Args:
            input_file: Path to input JSON file with articles
            output_file: Path to output JSON file for results
            max_tags: Maximum number of tags per article (default: 10)
        """
        print(f"Processing articles from {input_file}...")

        # Read input file
        articles = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

                # Try to parse as JSON array first
                if content.startswith('['):
                    articles = json.loads(content)
                else:
                    # Parse as JSONL (one JSON object per line)
                    for line in content.split('\n'):
                        line = line.strip()
                        if line:
                            try:
                                articles.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"Skipping invalid JSON line: {e}")

        except Exception as e:
            print(f"Error reading input file: {e}")
            return

        if not articles:
            print("No valid articles found in input file")
            return

        print(f"Found {len(articles)} articles to process")

        # Process each article
        results = []
        total = len(articles)

        for i, article in enumerate(articles):
            # Show progress
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"Processing article {i + 1}/{total}...")

            try:
                # Combine headline and short_description
                text = f"{article.get('headline', '')} {article.get('short_description', '')}"

                # Generate tags
                tags_with_weights = self.generate_tags_with_weights(text, max_tags)

                # Convert to dict format
                tags_dict = {tag: weight for tag, weight in tags_with_weights}

                # Create result entry
                result = {
                    "link": article.get('link', ''),
                    "tags": tags_dict
                }
                results.append(result)

            except Exception as e:
                print(f"Error processing article {i + 1}: {e}")
                # Add empty tags for failed articles
                result = {
                    "link": article.get('link', ''),
                    "tags": {}
                }
                results.append(result)

        # Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nProcessing complete! Results saved to {output_file}")
            print(f"Successfully processed {len(results)} articles")
        except Exception as e:
            print(f"Error saving output file: {e}")

if __name__ == "__main__":
    # technique = ['openai', 'groq']
    technique = ['groq']
    start_time = time.time()
    text = "If you only have 24 hours in a day, your success is dependent upon how you use the 24. You got to hear me. " \
           "People talk about Oprah Winfrey, you know, Ted Turner, Warren Buffett. Listen to me. I don't care how much" \
           " money you make, you only get 24 hours in a day. And the difference between Oprah and the person that's " \
           "broke is Oprah uses her 24 hours wisely. That's it. Listen to me. That's it. You get 24. I don't care. " \
           "You broke. You grew up broke. I don't care if you grew up rich. I don't care if you're in college. " \
           "You're not in college. You only get 24 hours. And I blew up. Literally, I went from being a high school " \
           "dropout to selling 6,000 books in less than six months. What happened? My 24 hours. I was like, okay, " \
           "Eric, you got to get a grip on your 24 hours because you about to be broke for the rest of your life. And " \
           "that's all I need you to do for me. I can tell you all about your life if you just Write down your 24 " \
           "hour schedule for me. You let me look at it. I can tell you where you're going to be in five years. I can " \
           "tell you where you're going to be in 10 years. I can tell you where you're going to be in 20 years if you keep that schedule."
    for tech in technique:
        generator = TagGenerator(technique=tech)
        tags = generator.generate_tags_with_weights(text, max_tags=10)

        print(f"\nTags generated using {tech}:")
        for tag, weight in tags:
            print(f"  {tag}: {weight:.3f}")

    total_time = time.time() - start_time
    print(f"\n\nTotal Execution time: {total_time:.3f} seconds")

    #Process Files
    # generator = TagGenerator(technique='groq')
    # start_time = time.time()
    # generator.process_articles_file(
    #     'input.json',
    #     'gorq_output.json',
    #     max_tags=10
    # )
    # total_time = time.time() - start_time
    # print(f"\n\nTotal Execution time: {total_time:.3f} seconds")

