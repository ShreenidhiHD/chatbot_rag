#!/usr/bin/env python3
"""
RAG Testing Script
Tests the RAG system with predefined questions and saves results.
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.rag import advanced_rag_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_questions(file_path: str) -> list:
    """Load test questions from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Split by double newlines (empty lines)
        questions = [q.strip() for q in content.split('\n\n') if q.strip()]
        
        # If no double newlines, split by single newlines
        if len(questions) == 1:
            questions = [q.strip() for q in content.split('\n') if q.strip()]
        
        logger.info(f"Loaded {len(questions)} test questions")
        return questions
        
    except FileNotFoundError:
        logger.error(f"Test questions file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading test questions: {e}")
        return []


def test_rag_system(
    questions: list,
    llm_provider: str = "local",
    save_results: bool = True,
    verbose: bool = True
) -> dict:
    """
    Test RAG system with list of questions.
    
    Args:
        questions: List of test questions
        llm_provider: LLM provider to use
        save_results: Whether to save results to file
        verbose: Whether to show detailed output
        
    Returns:
        Dictionary with test results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "llm_provider": llm_provider,
        "total_questions": len(questions),
        "successful_responses": 0,
        "failed_responses": 0,
        "intent_distribution": {},
        "agent_distribution": {},
        "test_results": []
    }
    
    logger.info(f"Starting RAG testing with {len(questions)} questions using {llm_provider} LLM")
    
    for i, question in enumerate(questions, 1):
        try:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Question {i}/{len(questions)}")
                print(f"{'='*60}")
                print(f"Q: {question}")
                print("-" * 60)
            
            # Call RAG system with intelligent context decisions
            result = advanced_rag_query(
                query=question,
                llm_provider=llm_provider
            )
            
            if "error" not in result:
                results["successful_responses"] += 1
                
                # Track intent distribution
                intent = result.get("intent", "unknown")
                results["intent_distribution"][intent] = results["intent_distribution"].get(intent, 0) + 1
                
                # Track agent distribution
                agent = result.get("agent", "unknown")
                results["agent_distribution"][agent] = results["agent_distribution"].get(agent, 0) + 1
                
                if verbose:
                    print(f"Intent: {intent}")
                    print(f"Agent: {agent}")
                    print(f"Sources: {len(result.get('sources', []))}")
                    print(f"A: {result['response']}")
                
            else:
                results["failed_responses"] += 1
                if verbose:
                    print(f"ERROR: {result.get('error', 'Unknown error')}")
            
            # Store individual result
            test_result = {
                "question_number": i,
                "question": question,
                "intent": result.get("intent", "error"),
                "agent": result.get("agent", "error"),
                "response": result.get("response", ""),
                "sources_count": len(result.get("sources", [])),
                "error": result.get("error", None)
            }
            results["test_results"].append(test_result)
            
        except Exception as e:
            results["failed_responses"] += 1
            logger.error(f"Error processing question {i}: {e}")
            
            if verbose:
                print(f"EXCEPTION: {e}")
            
            # Store error result
            test_result = {
                "question_number": i,
                "question": question,
                "intent": "error",
                "agent": "error",
                "response": "",
                "sources_count": 0,
                "error": str(e)
            }
            results["test_results"].append(test_result)
    
    # Calculate success rate
    total = results["total_questions"]
    success = results["successful_responses"]
    results["success_rate"] = (success / total * 100) if total > 0 else 0
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Questions: {total}")
        print(f"Successful: {success}")
        print(f"Failed: {results['failed_responses']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"\nIntent Distribution:")
        for intent, count in results["intent_distribution"].items():
            print(f"  {intent}: {count}")
        print(f"\nAgent Distribution:")
        for agent, count in results["agent_distribution"].items():
            print(f"  {agent}: {count}")
    
    # Save results to file
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"rag_test_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    return results


def save_formatted_results(results: dict, filename: str = None):
    """Save results in Q&A format"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_test_qa_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"RAG System Test Results\n")
            f.write(f"Generated: {results['timestamp']}\n")
            f.write(f"LLM Provider: {results['llm_provider']}\n")
            f.write(f"Success Rate: {results['success_rate']:.1f}%\n")
            f.write("=" * 80 + "\n\n")
            
            for test_result in results["test_results"]:
                f.write(f"Q{test_result['question_number']}: {test_result['question']}\n")
                f.write(f"Intent: {test_result['intent']} | Agent: {test_result['agent']}\n")
                if test_result['error']:
                    f.write(f"ERROR: {test_result['error']}\n")
                else:
                    f.write(f"A: {test_result['response']}\n")
                f.write("\n" + "-" * 80 + "\n\n")
        
        logger.info(f"Formatted results saved to: {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save formatted results: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG system with predefined questions")
    parser.add_argument(
        "--questions-file",
        default="core/scripts/test_questions.txt",
        help="Path to test questions file"
    )
    parser.add_argument(
        "--llm-provider",
        default="local",
        choices=["local", "openai", "gemini"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    # Load test questions
    questions = load_test_questions(args.questions_file)
    if not questions:
        logger.error("No test questions loaded. Exiting.")
        sys.exit(1)
    
    # Run tests
    try:
        results = test_rag_system(
            questions=questions,
            llm_provider=args.llm_provider,
            save_results=not args.no_save,
            verbose=not args.quiet
        )
        
        # Save formatted results
        if not args.no_save:
            save_formatted_results(results)
        
        logger.info("RAG testing completed successfully")
        
    except Exception as e:
        logger.error(f"RAG testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
