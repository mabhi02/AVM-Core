from typing import List, Dict, Set, Tuple
import re
import sympy
from sympy.parsing.latex import parse_latex
import networkx as nx

class SymbolProcessor:
    """Processes mathematical symbols and their relationships"""
    def __init__(self):
        self.symbol_types = {
            'variable': r'[a-zA-Z][_\d]*',
            'number': r'\d+\.?\d*',
            'operator': r'[+\-*/=<>≤≥≠∈∉⊆⊂∪∩]',
            'grouping': r'[(){}[\]]',
            'greek': r'\\[a-zA-Z]+',
            'function': r'\\(?:sin|cos|tan|log|ln|exp|lim|sup|inf|max|min)'
        }
        
        self.relation_types = {
            'equals': '=',
            'less_than': '<',
            'greater_than': '>',
            'leq': '≤',
            'geq': '≥',
            'in': '∈',
            'subset': '⊆',
            'union': '∪',
            'intersection': '∩'
        }
        
    def process_symbols(self, expression: str) -> Dict:
        """
        Process mathematical expression to extract and classify symbols
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Dictionary containing classified symbols and their properties
        """
        try:
            # Clean expression
            clean_expr = self._clean_expression(expression)
            
            # Extract and classify symbols
            symbols = self._extract_symbols(clean_expr)
            
            # Parse expression structure
            structure = self._parse_structure(clean_expr)
            
            # Extract relationships
            relations = self._extract_relations(clean_expr, symbols)
            
            # Build symbol graph
            graph = self._build_symbol_graph(symbols, relations)
            
            return {
                'symbols': symbols,
                'structure': structure,
                'relations': relations,
                'graph': graph,
                'complexity': self._calculate_complexity(symbols, relations)
            }
            
        except Exception as e:
            raise ValueError(f"Error processing symbols: {str(e)}")
    
    def _clean_expression(self, expression: str) -> str:
        """Clean and normalize mathematical expression"""
        # Remove whitespace
        expr = re.sub(r'\s+', '', expression)
        
        # Normalize operators
        replacements = {
            '\\leq': '≤',
            '\\geq': '≥',
            '\\neq': '≠',
            '\\in': '∈',
            '\\subset': '⊂',
            '\\subseteq': '⊆',
            '\\cup': '∪',
            '\\cap': '∩'
        }
        
        for tex, symbol in replacements.items():
            expr = expr.replace(tex, symbol)
            
        return expr
    
    def _extract_symbols(self, expression: str) -> Dict[str, List[str]]:
        """Extract and classify symbols from expression"""
        symbols = {category: [] for category in self.symbol_types}
        
        for category, pattern in self.symbol_types.items():
            matches = re.finditer(pattern, expression)
            for match in matches:
                symbol = match.group()
                if symbol not in symbols[category]:
                    symbols[category].append(symbol)
                    
        return symbols
    
    def _parse_structure(self, expression: str) -> Dict:
        """Parse the mathematical structure of the expression"""
        try:
            # Try to parse with sympy
            parsed = parse_latex(expression)
            
            return {
                'type': str(type(parsed)),
                'structure': str(parsed),
                'components': [str(arg) for arg in parsed.args],
                'variables': [str(sym) for sym in parsed.free_symbols]
            }
        except:
            # Fallback to basic structure analysis
            return self._basic_structure_analysis(expression)
    
    def _basic_structure_analysis(self, expression: str) -> Dict:
        """Basic analysis of expression structure"""
        structure = {
            'depth': 0,
            'components': [],
            'operators': []
        }
        
        current_depth = 0
        current_component = ''
        
        for char in expression:
            if char in '({[':
                current_depth += 1
                structure['depth'] = max(structure['depth'], current_depth)
            elif char in ')}]':
                current_depth -= 1
            elif char in self.relation_types.values():
                if current_component:
                    structure['components'].append(current_component)
                    current_component = ''
                structure['operators'].append(char)
            else:
                current_component += char
                
        if current_component:
            structure['components'].append(current_component)
            
        return structure
    
    def _extract_relations(self, expression: str, symbols: Dict[str, List[str]]) -> List[Dict]:
        """Extract relationships between symbols"""
        relations = []
        
        # Find all relation operators
        for rel_name, rel_symbol in self.relation_types.items():
            positions = [m.start() for m in re.finditer(re.escape(rel_symbol), expression)]
            
            for pos in positions:
                # Extract left and right parts
                left = expression[:pos].rstrip()
                right = expression[pos+1:].lstrip()
                
                relations.append({
                    'type': rel_name,
                    'symbol': rel_symbol,
                    'left': left,
                    'right': right,
                    'position': pos
                })
                
        return relations
    
    def _build_symbol_graph(self, symbols: Dict[str, List[str]], 
                          relations: List[Dict]) -> nx.Graph:
        """Build a graph representing symbol relationships"""
        G = nx.Graph()
        
        # Add nodes for all symbols
        for category, symbol_list in symbols.items():
            for symbol in symbol_list:
                G.add_node(symbol, type=category)
                
        # Add edges for relations
        for relation in relations:
            G.add_edge(
                relation['left'],
                relation['right'],
                type=relation['type']
            )
            
        return G
    
    def _calculate_complexity(self, symbols: Dict[str, List[str]], 
                            relations: List[Dict]) -> float:
        """Calculate complexity score for the expression"""
        factors = [
            len(relations),  # Number of relations
            sum(len(syms) for syms in symbols.values()),  # Total symbols
            len(symbols['operator']) if 'operator' in symbols else 0,  # Operators
            len(symbols['function']) if 'function' in symbols else 0,  # Functions
        ]


        # Normalize to 0-1 range
        complexity = sum(factors) / (10 * max(1, max(factors)))
        return min(1.0, complexity)

    def analyze_dependencies(self, symbols: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """Analyze dependencies between symbols"""
        dependencies = {}
        
        # Check for subscript dependencies
        for category, symbol_list in symbols.items():
            for symbol in symbol_list:
                if '_' in symbol:
                    base, subscript = symbol.split('_')
                    if base not in dependencies:
                        dependencies[base] = set()
                    dependencies[base].add(subscript)
        
        return dependencies

    def merge_symbol_sets(self, set1: Dict, set2: Dict) -> Dict:
        """Merge two sets of symbols, maintaining categories"""
        merged = {category: [] for category in self.symbol_types}
        
        # Merge symbols from both sets
        for category in self.symbol_types:
            symbols1 = set1.get(category, [])
            symbols2 = set2.get(category, [])
            merged[category] = sorted(set(symbols1 + symbols2))
            
        return merged

    def get_symbol_context(self, symbol: str, expression: str, window: int = 10) -> str:
        """Get the context around a symbol in an expression"""
        try:
            pos = expression.index(symbol)
            start = max(0, pos - window)
            end = min(len(expression), pos + len(symbol) + window)
            return expression[start:end]
        except ValueError:
            return ""

    def classify_symbol_role(self, symbol: str, expression: str) -> str:
        """Classify the mathematical role of a symbol"""
        context = self.get_symbol_context(symbol, expression)
        
        # Check for common mathematical roles
        if any(f in context for f in ['∀', '\\forall']):
            return 'quantifier_variable'
        elif any(f in context for f in ['∃', '\\exists']):
            return 'existential_variable'
        elif any(f in context for f in ['∑', '\\sum', '∏', '\\prod']):
            return 'index_variable'
        elif any(f in context for f in ['→', '\\to', '↦', '\\mapsto']):
            return 'function_variable'
        elif symbol in self.symbol_types['greek']:
            return 'parameter'
        else:
            return 'general_variable'

    def extract_functions(self, expression: str) -> List[Dict[str, str]]:
        """Extract function definitions and their properties"""
        functions = []
        
        # Match function patterns
        function_pattern = r'([a-zA-Z][_\d]*)\s*:\s*([^\s]+)\s*(?:→|\\to)\s*([^\s]+)'
        matches = re.finditer(function_pattern, expression)
        
        for match in matches:
            functions.append({
                'name': match.group(1),
                'domain': match.group(2),
                'codomain': match.group(3)
            })
            
        return functions

    def validate_symbol_usage(self, symbols: Dict[str, List[str]], 
                            expression: str) -> List[str]:
        """Validate correct usage of mathematical symbols"""
        errors = []
        
        # Check for undefined symbols
        all_defined_symbols = set().union(*[set(syms) for syms in symbols.values()])
        used_symbols = set(re.findall(r'[a-zA-Z][_\d]*', expression))
        
        undefined = used_symbols - all_defined_symbols
        if undefined:
            errors.append(f"Undefined symbols: {', '.join(undefined)}")
        
        # Check for consistent usage of operators
        for op in symbols.get('operator', []):
            if op in expression:
                if not self._check_operator_usage(op, expression):
                    errors.append(f"Invalid usage of operator: {op}")
        
        # Check grouping symbols
        if not self._check_balanced_grouping(expression):
            errors.append("Unbalanced grouping symbols")
            
        return errors

    def _check_operator_usage(self, operator: str, expression: str) -> bool:
        """Check if an operator is used correctly"""
        # Get positions of operator
        positions = [m.start() for m in re.finditer(re.escape(operator), expression)]
        
        for pos in positions:
            # Check if operator has valid operands
            if pos == 0 or pos == len(expression) - 1:
                return False
                
            # Check surrounding characters
            prev_char = expression[pos - 1]
            next_char = expression[pos + 1]
            
            if prev_char in self.relation_types.values() or next_char in self.relation_types.values():
                return False
                
        return True

    def _check_balanced_grouping(self, expression: str) -> bool:
        """Check if grouping symbols are balanced"""
        stack = []
        pairs = {')': '(', '}': '{', ']': '['}
        
        for char in expression:
            if char in '({[':
                stack.append(char)
            elif char in ')}]':
                if not stack or stack.pop() != pairs[char]:
                    return False
                    
        return len(stack) == 0

    def create_symbol_vocabulary(self, expressions: List[str]) -> Dict[str, int]:
        """Create a vocabulary of all symbols with their frequencies"""
        vocab = {}
        
        for expr in expressions:
            symbols = self._extract_symbols(expr)
            for category, symbol_list in symbols.items():
                for symbol in symbol_list:
                    if symbol not in vocab:
                        vocab[symbol] = 0
                    vocab[symbol] += 1
                    
        return dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))

    def get_symbol_substitutions(self, expression: str) -> Dict[str, List[str]]:
        """Get possible symbol substitutions maintaining mathematical meaning"""
        substitutions = {}
        symbols = self._extract_symbols(expression)
        
        for category, symbol_list in symbols.items():
            for symbol in symbol_list:
                role = self.classify_symbol_role(symbol, expression)
                substitutions[symbol] = self._get_substitution_options(role)
                
        return substitutions

    def _get_substitution_options(self, role: str) -> List[str]:
        """Get substitution options based on symbol role"""
        options = {
            'quantifier_variable': ['x', 'y', 'z', 'n', 'm'],
            'existential_variable': ['a', 'b', 'c', 'p', 'q'],
            'index_variable': ['i', 'j', 'k', 'l', 'n'],
            'function_variable': ['f', 'g', 'h', 'φ', 'ψ'],
            'parameter': ['α', 'β', 'γ', 'λ', 'μ'],
            'general_variable': ['u', 'v', 'w', 'r', 's']
        }
        return options.get(role, [])

    def to_latex(self, symbols: Dict[str, List[str]], 
                relations: List[Dict]) -> str:
        """Convert symbols and relations to LaTeX format"""
        latex = ""
        
        # Convert symbols
        for category, symbol_list in symbols.items():
            if symbol_list:
                latex += f"% {category} symbols\n"
                for symbol in symbol_list:
                    latex += f"{self._symbol_to_latex(symbol, category)}\n"
                    
        # Convert relations
        if relations:
            latex += "% Relations\n"
            for relation in relations:
                latex += f"{relation['left']} {self._relation_to_latex(relation['symbol'])} {relation['right']}\n"
                
        return latex

    def _symbol_to_latex(self, symbol: str, category: str) -> str:
        """Convert a symbol to its LaTeX representation"""
        if category == 'greek':
            return f"\\{symbol}"
        elif category == 'function':
            return f"\\mathrm{{{symbol}}}"
        elif '_' in symbol:
            base, sub = symbol.split('_')
            return f"{base}_{{{sub}}}"
        else:
            return symbol

    def _relation_to_latex(self, relation: str) -> str:
        """Convert a relation symbol to LaTeX"""
        latex_relations = {
            '≤': '\\leq',
            '≥': '\\geq',
            '≠': '\\neq',
            '∈': '\\in',
            '⊆': '\\subseteq',
            '⊂': '\\subset',
            '∪': '\\cup',
            '∩': '\\cap'
        }
        return latex_relations.get(relation, relation)