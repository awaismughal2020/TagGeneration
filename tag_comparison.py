import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from typing import List, Dict, Tuple, Set
import warnings

warnings.filterwarnings('ignore')


class TagComparisonAnalyzer:
    """
    Comprehensive analyzer for comparing OpenAI vs spaCy tag generation quality
    """

    def __init__(self, openai_file: str, spacy_file: str):
        """Initialize with file paths"""
        self.openai_data = self.load_data(openai_file, "OpenAI")
        self.spacy_data = self.load_data(spacy_file, "spaCy")
        self.results = {}

    def load_data(self, filepath: str, source: str) -> pd.DataFrame:
        """Load and validate JSON data"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            df['source'] = source

            # Ensure tags are lists
            df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])

            print(f"✅ Loaded {len(df)} articles from {source}")
            return df

        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return pd.DataFrame()

    def basic_statistics(self) -> Dict:
        """Calculate basic tag statistics"""
        stats = {}

        for df, name in [(self.openai_data, 'OpenAI'), (self.spacy_data, 'spaCy')]:
            if df.empty:
                continue

            tag_counts = df['tags'].apply(len)
            all_tags = [tag for tags in df['tags'] for tag in tags]
            unique_tags = set(all_tags)

            stats[name] = {
                'total_articles': len(df),
                'avg_tags_per_article': tag_counts.mean(),
                'median_tags_per_article': tag_counts.median(),
                'std_tags_per_article': tag_counts.std(),
                'min_tags': tag_counts.min(),
                'max_tags': tag_counts.max(),
                'total_tags_generated': len(all_tags),
                'unique_tags': len(unique_tags),
                'tag_diversity_ratio': len(unique_tags) / len(all_tags) if all_tags else 0,
                'most_common_tags': Counter(all_tags).most_common(10)
            }

        self.results['basic_stats'] = stats
        return stats

    def tag_overlap_analysis(self) -> Dict:
        """Analyze tag overlap between OpenAI and spaCy for same articles"""
        overlap_metrics = {
            'jaccard_similarities': [],
            'overlap_coefficients': [],
            'common_tag_counts': [],
            'union_tag_counts': []
        }

        min_len = min(len(self.openai_data), len(self.spacy_data))

        for i in range(min_len):
            openai_tags = set(self.openai_data.iloc[i]['tags'])
            spacy_tags = set(self.spacy_data.iloc[i]['tags'])

            intersection = openai_tags.intersection(spacy_tags)
            union = openai_tags.union(spacy_tags)

            # Jaccard similarity
            jaccard = len(intersection) / len(union) if union else 0

            # Overlap coefficient (Szymkiewicz–Simpson coefficient)
            overlap_coeff = len(intersection) / min(len(openai_tags),
                                                    len(spacy_tags)) if openai_tags and spacy_tags else 0

            overlap_metrics['jaccard_similarities'].append(jaccard)
            overlap_metrics['overlap_coefficients'].append(overlap_coeff)
            overlap_metrics['common_tag_counts'].append(len(intersection))
            overlap_metrics['union_tag_counts'].append(len(union))

        summary = {
            'avg_jaccard_similarity': np.mean(overlap_metrics['jaccard_similarities']),
            'avg_overlap_coefficient': np.mean(overlap_metrics['overlap_coefficients']),
            'avg_common_tags': np.mean(overlap_metrics['common_tag_counts']),
            'articles_with_no_overlap': sum(1 for x in overlap_metrics['common_tag_counts'] if x == 0),
            'articles_with_high_overlap': sum(1 for x in overlap_metrics['jaccard_similarities'] if x > 0.5)
        }

        self.results['overlap_analysis'] = {
            'metrics': overlap_metrics,
            'summary': summary
        }
        return self.results['overlap_analysis']

    def content_relevance_analysis(self) -> Dict:
        """Analyze how relevant tags are to article content"""
        relevance_scores = {'OpenAI': [], 'spaCy': []}

        for df, name in [(self.openai_data, 'OpenAI'), (self.spacy_data, 'spaCy')]:
            if df.empty:
                continue

            for _, row in df.iterrows():
                # Combine headline and short_description
                content = f"{row['headline']} {row['short_description']}"
                tags_text = " ".join(row['tags'])

                if not tags_text.strip():
                    relevance_scores[name].append(0)
                    continue

                # Calculate TF-IDF similarity
                vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
                try:
                    tfidf_matrix = vectorizer.fit_transform([content, tags_text])
                    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                    relevance_scores[name].append(similarity)
                except:
                    relevance_scores[name].append(0)

        summary = {}
        for name, scores in relevance_scores.items():
            if scores:
                summary[name] = {
                    'avg_relevance_score': np.mean(scores),
                    'median_relevance_score': np.median(scores),
                    'std_relevance_score': np.std(scores),
                    'high_relevance_articles': sum(1 for s in scores if s > 0.3)
                }

        self.results['content_relevance'] = summary
        return summary

    def tag_quality_metrics(self) -> Dict:
        """Analyze various tag quality metrics"""
        quality_metrics = {}

        for df, name in [(self.openai_data, 'OpenAI'), (self.spacy_data, 'spaCy')]:
            if df.empty:
                continue

            all_tags = [tag for tags in df['tags'] for tag in tags]

            # Tag length distribution
            tag_lengths = [len(tag) for tag in all_tags]

            # Check for meaningful tags (not just single characters or numbers)
            meaningful_tags = [tag for tag in all_tags if len(tag) > 2 and not tag.isdigit()]

            # Check for proper capitalization
            properly_capitalized = [tag for tag in all_tags if tag.istitle() or tag.islower()]

            # Check for stop words
            stop_word_tags = [tag for tag in all_tags if tag.lower() in ENGLISH_STOP_WORDS]

            quality_metrics[name] = {
                'avg_tag_length': np.mean(tag_lengths) if tag_lengths else 0,
                'meaningful_tag_ratio': len(meaningful_tags) / len(all_tags) if all_tags else 0,
                'proper_capitalization_ratio': len(properly_capitalized) / len(all_tags) if all_tags else 0,
                'stop_word_ratio': len(stop_word_tags) / len(all_tags) if all_tags else 0,
                'single_word_tags': sum(1 for tag in all_tags if len(tag.split()) == 1),
                'multi_word_tags': sum(1 for tag in all_tags if len(tag.split()) > 1)
            }

        self.results['quality_metrics'] = quality_metrics
        return quality_metrics

    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('OpenAI vs spaCy Tag Generation Comparison', fontsize=16, fontweight='bold')

        # 1. Tags per article distribution
        openai_tag_counts = self.openai_data['tags'].apply(len)
        spacy_tag_counts = self.spacy_data['tags'].apply(len)

        axes[0, 0].hist([openai_tag_counts, spacy_tag_counts], bins=20, alpha=0.7,
                        label=['OpenAI', 'spaCy'], color=['blue', 'red'])
        axes[0, 0].set_title('Distribution of Tags per Article')
        axes[0, 0].set_xlabel('Number of Tags')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()

        # 2. Average tags per article comparison
        avg_tags = [self.results['basic_stats']['OpenAI']['avg_tags_per_article'],
                    self.results['basic_stats']['spaCy']['avg_tags_per_article']]
        axes[0, 1].bar(['OpenAI', 'spaCy'], avg_tags, color=['blue', 'red'], alpha=0.7)
        axes[0, 1].set_title('Average Tags per Article')
        axes[0, 1].set_ylabel('Average Number of Tags')

        # 3. Tag diversity comparison
        diversity = [self.results['basic_stats']['OpenAI']['tag_diversity_ratio'],
                     self.results['basic_stats']['spaCy']['tag_diversity_ratio']]
        axes[0, 2].bar(['OpenAI', 'spaCy'], diversity, color=['blue', 'red'], alpha=0.7)
        axes[0, 2].set_title('Tag Diversity Ratio')
        axes[0, 2].set_ylabel('Unique Tags / Total Tags')

        # 4. Jaccard similarity distribution
        if 'overlap_analysis' in self.results:
            jaccard_sims = self.results['overlap_analysis']['metrics']['jaccard_similarities']
            axes[1, 0].hist(jaccard_sims, bins=20, alpha=0.7, color='green')
            axes[1, 0].set_title('Jaccard Similarity Distribution')
            axes[1, 0].set_xlabel('Jaccard Similarity')
            axes[1, 0].set_ylabel('Frequency')

        # 5. Content relevance comparison
        if 'content_relevance' in self.results:
            relevance_openai = self.results['content_relevance']['OpenAI']['avg_relevance_score']
            relevance_spacy = self.results['content_relevance']['spaCy']['avg_relevance_score']
            axes[1, 1].bar(['OpenAI', 'spaCy'], [relevance_openai, relevance_spacy],
                           color=['blue', 'red'], alpha=0.7)
            axes[1, 1].set_title('Average Content Relevance Score')
            axes[1, 1].set_ylabel('TF-IDF Similarity Score')

        # 6. Tag quality metrics
        if 'quality_metrics' in self.results:
            meaningful_ratios = [self.results['quality_metrics']['OpenAI']['meaningful_tag_ratio'],
                                 self.results['quality_metrics']['spaCy']['meaningful_tag_ratio']]
            axes[1, 2].bar(['OpenAI', 'spaCy'], meaningful_ratios, color=['blue', 'red'], alpha=0.7)
            axes[1, 2].set_title('Meaningful Tag Ratio')
            axes[1, 2].set_ylabel('Ratio')

        # 7. Tag length comparison
        openai_tags = [tag for tags in self.openai_data['tags'] for tag in tags]
        spacy_tags = [tag for tags in self.spacy_data['tags'] for tag in tags]
        openai_lengths = [len(tag) for tag in openai_tags]
        spacy_lengths = [len(tag) for tag in spacy_tags]

        axes[2, 0].boxplot([openai_lengths, spacy_lengths], labels=['OpenAI', 'spaCy'])
        axes[2, 0].set_title('Tag Length Distribution')
        axes[2, 0].set_ylabel('Tag Length (characters)')

        # 8. Most common tags word cloud style visualization
        if 'basic_stats' in self.results:
            openai_common = dict(self.results['basic_stats']['OpenAI']['most_common_tags'])
            spacy_common = dict(self.results['basic_stats']['spaCy']['most_common_tags'])

            # Bar plot of top 10 tags for each
            axes[2, 1].barh(range(len(openai_common)), list(openai_common.values()),
                            alpha=0.7, color='blue', label='OpenAI')
            axes[2, 1].set_yticks(range(len(openai_common)))
            axes[2, 1].set_yticklabels(list(openai_common.keys()))
            axes[2, 1].set_title('OpenAI Top 10 Tags')
            axes[2, 1].set_xlabel('Frequency')

        # 9. Overlap metrics summary
        if 'overlap_analysis' in self.results:
            overlap_summary = self.results['overlap_analysis']['summary']
            metrics = ['Avg Jaccard', 'Avg Overlap Coeff', 'Avg Common Tags']
            values = [overlap_summary['avg_jaccard_similarity'],
                      overlap_summary['avg_overlap_coefficient'],
                      overlap_summary['avg_common_tags'] / 10]  # Scale for visibility

            axes[2, 2].bar(metrics, values, color=['purple', 'orange', 'green'], alpha=0.7)
            axes[2, 2].set_title('Tag Overlap Summary')
            axes[2, 2].set_ylabel('Score')
            axes[2, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def generate_detailed_report(self) -> str:
        """Generate a comprehensive comparison report"""
        report = []
        report.append("=" * 80)
        report.append("OPENAI vs SPACY TAG GENERATION COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")

        # Executive Summary
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("-" * 40)

        if 'basic_stats' in self.results:
            openai_stats = self.results['basic_stats']['OpenAI']
            spacy_stats = self.results['basic_stats']['spaCy']

            report.append(f"📈 Average Tags per Article:")
            report.append(f"   • OpenAI: {openai_stats['avg_tags_per_article']:.2f}")
            report.append(f"   • spaCy:  {spacy_stats['avg_tags_per_article']:.2f}")
            report.append("")

            report.append(f"🎯 Tag Diversity (Uniqueness):")
            report.append(f"   • OpenAI: {openai_stats['tag_diversity_ratio']:.3f}")
            report.append(f"   • spaCy:  {spacy_stats['tag_diversity_ratio']:.3f}")
            report.append("")

        if 'content_relevance' in self.results:
            report.append(f"🔗 Content Relevance Score:")
            report.append(f"   • OpenAI: {self.results['content_relevance']['OpenAI']['avg_relevance_score']:.3f}")
            report.append(f"   • spaCy:  {self.results['content_relevance']['spaCy']['avg_relevance_score']:.3f}")
            report.append("")

        # Detailed Analysis
        report.append("🔍 DETAILED ANALYSIS")
        report.append("-" * 40)

        # Tag Overlap Analysis
        if 'overlap_analysis' in self.results:
            overlap = self.results['overlap_analysis']['summary']
            report.append("🔄 Tag Overlap Between OpenAI and spaCy:")
            report.append(f"   • Average Jaccard Similarity: {overlap['avg_jaccard_similarity']:.3f}")
            report.append(f"   • Average Overlap Coefficient: {overlap['avg_overlap_coefficient']:.3f}")
            report.append(f"   • Articles with No Tag Overlap: {overlap['articles_with_no_overlap']}")
            report.append(f"   • Articles with High Overlap (>50%): {overlap['articles_with_high_overlap']}")
            report.append("")

        # Quality Metrics
        if 'quality_metrics' in self.results:
            report.append("✨ Tag Quality Metrics:")
            for tool in ['OpenAI', 'spaCy']:
                metrics = self.results['quality_metrics'][tool]
                report.append(f"   {tool}:")
                report.append(f"     • Average Tag Length: {metrics['avg_tag_length']:.1f} characters")
                report.append(f"     • Meaningful Tags: {metrics['meaningful_tag_ratio']:.1%}")
                report.append(f"     • Proper Capitalization: {metrics['proper_capitalization_ratio']:.1%}")
                report.append(f"     • Stop Word Ratio: {metrics['stop_word_ratio']:.1%}")
                report.append(f"     • Multi-word Tags: {metrics['multi_word_tags']}")
            report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)

        # Determine winner based on multiple criteria
        scores = {'OpenAI': 0, 'spaCy': 0}

        if 'basic_stats' in self.results:
            # Higher diversity is better
            if openai_stats['tag_diversity_ratio'] > spacy_stats['tag_diversity_ratio']:
                scores['OpenAI'] += 1
            else:
                scores['spaCy'] += 1

        if 'content_relevance' in self.results:
            # Higher relevance is better
            if (self.results['content_relevance']['OpenAI']['avg_relevance_score'] >
                    self.results['content_relevance']['spaCy']['avg_relevance_score']):
                scores['OpenAI'] += 1
            else:
                scores['spaCy'] += 1

        if 'quality_metrics' in self.results:
            # Higher meaningful tag ratio is better
            if (self.results['quality_metrics']['OpenAI']['meaningful_tag_ratio'] >
                    self.results['quality_metrics']['spaCy']['meaningful_tag_ratio']):
                scores['OpenAI'] += 1
            else:
                scores['spaCy'] += 1

        winner = max(scores, key=scores.get)
        report.append(f"🏆 OVERALL WINNER: {winner}")
        report.append(f"   Score: OpenAI ({scores['OpenAI']}) vs spaCy ({scores['spaCy']})")
        report.append("")

        # Specific recommendations
        report.append("📝 Specific Recommendations:")
        report.append("   • For higher tag diversity and uniqueness → Choose the tool with higher diversity ratio")
        report.append("   • For better content relevance → Choose the tool with higher relevance scores")
        report.append("   • For production use → Consider hybrid approach using both tools")
        report.append("   • For cost efficiency → Consider the computational cost vs quality trade-off")

        return "\n".join(report)

    def run_complete_analysis(self):
        """Run all analyses and generate complete insights"""
        print("🚀 Starting comprehensive tag generation comparison...")
        print()

        # Run all analyses
        self.basic_statistics()
        print("✅ Basic statistics calculated")

        self.tag_overlap_analysis()
        print("✅ Tag overlap analysis completed")

        self.content_relevance_analysis()
        print("✅ Content relevance analysis completed")

        self.tag_quality_metrics()
        print("✅ Tag quality metrics calculated")

        # Generate visualizations
        print("📊 Generating visualizations...")
        self.generate_visualizations()

        # Generate and print detailed report
        print("\n" + "=" * 80)
        report = self.generate_detailed_report()
        print(report)

        return self.results


# Main execution
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = TagComparisonAnalyzer("openai_output.json", "spacy_output.json")

    # Run complete analysis
    results = analyzer.run_complete_analysis()

    # Save results to JSON for further analysis
    with open("tag_comparison_results.json", "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj


        import json

        json.dump(results, f, indent=2, default=convert_numpy)

    print("\n💾 Results saved to 'tag_comparison_results.json'")
    print("🎉 Analysis complete! Check the visualizations and report above for insights.")
