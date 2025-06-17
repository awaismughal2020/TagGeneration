import json
import os
import re
from typing import List, Dict, Any
import argparse
from dotenv import load_dotenv

load_dotenv()

# OpenAI imports
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Install with: pip install openai")

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
        self.technique = technique.lower()

        if self.technique == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library is required for openai technique")
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY environment variable is required")

        elif self.technique == "spacy":
            if not SPACY_AVAILABLE:
                raise ImportError("spaCy library is required for spacy technique")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy English model not found. Download with: python -m spacy download en_core_web_sm")
                raise

        else:
            raise ValueError("technique must be either 'openai' or 'spacy'")

    def generate_tags_openai(self, headline: str, short_description: str) -> List[str]:
        """Generate tags using OpenAI API"""
        prompt = f"""
        Based on the following headline and description, generate 3-7 relevant tags that best categorize this article. 
        Focus on key topics, themes, entities, and concepts. Return only the tags as a comma-separated list.

        Headline: {headline}
        Description: {short_description}

        Tags:
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that generates relevant tags for news articles. Return only comma-separated tags without explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )

            tags_text = response.choices[0].message.content.strip()
            # Clean and split tags
            tags = [tag.strip().lower() for tag in tags_text.split(',')]
            # Remove empty tags and duplicates
            tags = list(set([tag for tag in tags if tag and len(tag) > 1]))
            return tags

        except Exception as e:
            print(f"Error generating tags with OpenAI: {e}")
            return []

    def generate_tags_spacy(self, headline: str, short_description: str) -> List[str]:
        """Generate tags using spaCy NLP"""
        combined_text = f"{headline} {short_description}"
        doc = self.nlp(combined_text)

        tags = set()

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART"]:
                clean_entity = self._clean_tag(ent.text)
                if clean_entity:
                    tags.add(clean_entity)

        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Keep short noun phrases
                clean_chunk = self._clean_tag(chunk.text)
                if clean_chunk and clean_chunk not in STOP_WORDS:
                    tags.add(clean_chunk)

        # Extract important single nouns
        for token in doc:
            if (token.pos_ == "NOUN" and
                    not token.is_stop and
                    not token.is_punct and
                    len(token.text) > 2):
                clean_token = self._clean_tag(token.lemma_)
                if clean_token:
                    tags.add(clean_token)

        # Convert to list and limit number of tags
        tags_list = list(tags)[:8]  # Limit to 8 tags
        return tags_list

    def _clean_tag(self, tag: str) -> str:
        """Clean and normalize tags"""
        # Remove special characters and extra whitespace
        cleaned = re.sub(r'[^\w\s-]', '', tag.strip().lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Skip very short tags or common words
        if len(cleaned) < 2 or cleaned in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her',
                                           'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man',
                                           'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let',
                                           'put', 'say', 'she', 'too', 'use']:
            return ""

        return cleaned

    def process_articles(self, input_file: str, output_file: str):
        """Process articles from input file and save results to output file"""
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Parse JSON lines or JSON array
        articles = []
        if content.startswith('['):
            # JSON array format
            articles = json.loads(content)
        else:
            # JSON lines format
            for line in content.split('\n'):
                if line.strip():
                    articles.append(json.loads(line))

        results = []

        for i, article in enumerate(articles):
            print(f"Processing article {i + 1}/{len(articles)}: {article['headline'][:50]}...")

            if self.technique == "openai":
                tags = self.generate_tags_openai(article['headline'], article['short_description'])
            else:
                tags = self.generate_tags_spacy(article['headline'], article['short_description'])

            result = {
                "headline": article['headline'],
                "short_description": article['short_description'],
                "tags": tags
            }
            results.append(result)

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Processing complete! Results saved to {output_file}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Generate tags for articles using OpenAI or spaCy')
    parser.add_argument('--technique', '-t', choices=['openai', 'spacy'], default='spacy',
                        help='Tag generation technique (default: spacy)')
    parser.add_argument('--input', '-i', default='adasdsdsa.json',
                        help='Input JSON file (default: adasdsdsa.json)')
    parser.add_argument('--output', '-o', default='output.json',
                        help='Output JSON file (default: output.json)')

    args = parser.parse_args()

    try:
        generator = TagGenerator(technique=args.technique)
        generator.process_articles(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    tag_generation_technique = "spacy"  #"openai" or "spacy"

    try:
        generator = TagGenerator(technique=tag_generation_technique)
        results = generator.process_articles("input.json", "spacy_output.json")

        print("\nSample results:")
        for result in results[:2]:
            print(f"\nHeadline: {result['headline']}")
            print(f"Tags: {', '.join(result['tags'])}")

    except Exception as e:
        print(f"Error: {e}")