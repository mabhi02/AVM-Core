import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
import time
import json
import os

class MathStackScraper:
    def __init__(self):
        self.base_url = "https://math.stackexchange.com"
        self.api_url = "https://api.stackexchange.com/2.3"
        self.api_key = "rl_zvprTsFW4db4fWyBMTAeHZnfv"
        
    def scrape_proofs(self, num_pages: int = 100) -> List[Dict]:
        """Scrape mathematical proofs from Math StackExchange"""
        proofs = []
        page = 1
        
        while page <= num_pages and len(proofs) < 1000:
            try:
                # Get questions tagged with 'proof-writing'
                questions = self._get_questions(page)
                
                if not questions:  # Check if we hit API limit
                    print(f"No questions returned on page {page}, possible rate limit. Waiting...")
                    time.sleep(30)  # Wait longer if we hit rate limit
                    continue
                
                for question in questions:
                    if self._is_proof_question(question):
                        proof = self._process_question(question)
                        if proof:
                            proofs.append(proof)
                            print(f"Collected proof {len(proofs)}: {proof['theorem'][:100]}...")
                            
                page += 1
                time.sleep(2)  # Respect rate limits
                
            except Exception as e:
                print(f"Error on page {page}: {e}")
                time.sleep(5)  # Wait a bit longer on error
                continue
                
        return proofs
    
    def _get_questions(self, page: int) -> List[Dict]:
        """Get questions from Math StackExchange API"""
        params = {
            'page': page,
            'pagesize': 100,
            'order': 'desc',
            'sort': 'votes',
            'tagged': 'proof-writing',
            'site': 'math',
            'filter': 'withbody',
            'key': self.api_key
        }
        
        response = requests.get(f"{self.api_url}/questions", params=params)
        data = response.json()
        
        # Check remaining quota
        if 'quota_remaining' in data:
            print(f"API calls remaining: {data['quota_remaining']}")
            
        return data.get('items', [])
    
    def _is_proof_question(self, question: Dict) -> bool:
        """Check if question is asking for a proof"""
        title = question.get('title', '').lower()
        body = question.get('body', '').lower()
        
        proof_keywords = ['prove', 'proof', 'show that', 'demonstrate', 'verify', 'establish']
        return any(keyword in title or keyword in body for keyword in proof_keywords)
    
    def _process_question(self, question: Dict) -> Dict:
        """Process a question and its accepted answer into a proof"""
        # Get accepted answer or highest voted answer
        answer_id = question.get('accepted_answer_id')
        
        if not answer_id:
            # If no accepted answer, try to get highest voted answer
            answers = self._get_answers(question['question_id'])
            if answers:
                answer = max(answers, key=lambda x: x.get('score', 0))
            else:
                return None
        else:
            answer = self._get_answer(answer_id)
            
        if not answer:
            return None
            
        # Clean HTML content
        soup_question = BeautifulSoup(question.get('body', ''), 'html.parser')
        soup_answer = BeautifulSoup(answer.get('body', ''), 'html.parser')
        
        return {
            'theorem': question.get('title'),
            'context': soup_question.get_text(),
            'proof': soup_answer.get_text(),
            'tags': question.get('tags', []),
            'score': question.get('score', 0),
            'answer_score': answer.get('score', 0),
            'is_accepted': bool(question.get('accepted_answer_id') == answer.get('answer_id')),
            'question_id': question.get('question_id'),
            'answer_id': answer.get('answer_id')
        }
    
    def _get_answer(self, answer_id: int) -> Dict:
        """Get specific answer from API"""
        params = {
            'site': 'math',
            'filter': 'withbody',
            'key': self.api_key
        }
        response = requests.get(f"{self.api_url}/answers/{answer_id}", params=params)
        answers = response.json().get('items', [])
        return answers[0] if answers else None
    
    def _get_answers(self, question_id: int) -> List[Dict]:
        """Get all answers for a question"""
        params = {
            'site': 'math',
            'filter': 'withbody',
            'sort': 'votes',
            'order': 'desc',
            'key': self.api_key
        }
        response = requests.get(f"{self.api_url}/questions/{question_id}/answers", params=params)
        return response.json().get('items', [])

    def save_proofs(self, proofs: List[Dict], filename: str = 'data/math_proofs.json'):
        """Save collected proofs to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(proofs, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(proofs)} proofs to {filename}")

if __name__ == "__main__":
    print("Starting Math StackExchange proof scraper...")
    
    # Create data directory if it doesn't exist
    os.makedirs("../../data", exist_ok=True)
    
    # Initialize and run scraper
    scraper = MathStackScraper()
    print("Collecting proofs...")
    proofs = scraper.scrape_proofs(num_pages=100)  # Will collect up to 1000 proofs
    
    # Save the results
    output_file = "../../data/math_proofs.json"
    print(f"Saving {len(proofs)} proofs to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(proofs, f, ensure_ascii=False, indent=2)
    
    print("Scraping completed!")