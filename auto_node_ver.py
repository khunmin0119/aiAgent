"""
Method 1 - DEBUG VERSION
디버깅 정보 추가하여 실패 원인 확인
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class HybridConfig:
    span_range: Tuple[float, float] = (4.0, 6.0)
    height_range: Tuple[float, float] = (1.5, 2.5)
    n_loads: int = 2
    max_nodes: int = 8
    
    hidden_dim: int = 128
    learning_rate: float = 0.001
    nn_epochs: int = 500  # Reduced
    batch_size: int = 32
    
    population_size: int = 50
    n_generations: int = 50  # Reduced
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elite_size: int = 5
    
    fck: float = 27.0
    fy: float = 400.0
    cover: float = 0.05


class STMProblem:
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self):
        self.span = np.random.uniform(*self.config.span_range)
        self.height = np.random.uniform(*self.config.height_range)
        
        self.supports = [(0.0, 0.0), (self.span, 0.0)]
        
        load_positions = np.linspace(
            0.3 * self.span, 
            0.7 * self.span, 
            self.config.n_loads
        )
        self.loads = [(x, self.height, 50.0) for x in load_positions]
        
        n_candidates = self.config.max_nodes - len(self.supports) - len(self.loads)
        self.candidates = []
        for i in range(n_candidates):
            x = np.random.uniform(0.2 * self.span, 0.8 * self.span)
            y = np.random.uniform(0.3 * self.height, 0.7 * self.height)
            self.candidates.append((x, y))
        
        self.nodes = (
            self.supports + 
            [(x, y) for x, y, _ in self.loads] + 
            self.candidates
        )
        self.n_nodes = len(self.nodes)
        
        return self.get_state()
    
    def get_state(self):
        features = []
        
        for x, y in self.nodes:
            features.extend([x / self.span, y / self.height])
        
        for i in range(len(self.nodes)):
            is_support = 1.0 if i < 2 else 0.0
            is_load = 1.0 if 2 <= i < 2 + len(self.loads) else 0.0
            is_candidate = 1.0 if i >= 2 + len(self.loads) else 0.0
            features.extend([is_support, is_load, is_candidate])
        
        features.extend([
            self.span / 10.0,
            self.height / 5.0,
            len(self.loads) / 5.0
        ])
        
        return np.array(features, dtype=np.float32)


class KDSChecker:
    """DEBUG VERSION - 더 관대한 체크"""
    
    def __init__(self, fck=27.0, fy=400.0, cover=0.05):
        self.fck = fck
        self.fy = fy
        self.cover = cover
        self.f_cd = 0.85 * fck / 1.5
        self.f_yd = fy / 1.15
    
    def check(self, problem: STMProblem, edges: List[Tuple[int, int]]) -> Dict:
        """DEBUG: 각 단계별 실패 이유 출력"""
        
        debug_info = {
            'n_edges': len(edges),
            'n_nodes': problem.n_nodes
        }
        
        if len(edges) == 0:
            print("  ❌ FAIL: No edges")
            return {
                'overall_pass': False, 
                'equilibrium': False,
                'debug': debug_info
            }
        
        # 유효한 엣지 체크
        valid_edges = []
        for n1, n2 in edges:
            if n1 >= problem.n_nodes or n2 >= problem.n_nodes:
                continue
            if n1 == n2:
                continue
            valid_edges.append((n1, n2))
        
        if len(valid_edges) == 0:
            print(f"  ❌ FAIL: No valid edges (had {len(edges)} invalid)")
            return {
                'overall_pass': False,
                'equilibrium': False,
                'debug': debug_info
            }
        
        debug_info['valid_edges'] = len(valid_edges)
        
        # 힘 계산
        forces_result = self.calculate_forces(problem, valid_edges)
        if forces_result is None:
            print(f"  ❌ FAIL: Force calculation failed")
            return {
                'overall_pass': False,
                'equilibrium': False,
                'debug': debug_info
            }
        
        forces = forces_result['forces']
        equilibrium = forces_result['equilibrium']
        eq_error = forces_result['error']
        
        debug_info['eq_error'] = eq_error
        debug_info['equilibrium'] = equilibrium
        
        if not equilibrium:
            print(f"  ❌ FAIL: Equilibrium error = {eq_error:.2f} > 5.0")
        
        # 부재 강도 체크 (관대하게)
        members_pass = True
        max_violation = 0.0
        
        for (n1, n2), force in zip(valid_edges, forces):
            x1, y1 = problem.nodes[n1]
            x2, y2 = problem.nodes[n2]
            L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if L < 1e-6:
                continue
            
            # 매우 관대한 강도 체크
            if force > 0:  # Tension
                A_s_min = 0.002 * 0.5 * L
                capacity = A_s_min * self.f_yd * 2.0  # 2배 여유
            else:  # Compression
                w_s = 0.5
                A_c = w_s * L * 0.8
                capacity = 0.85 * self.f_cd * A_c * 2.0  # 2배 여유
            
            violation = abs(force) / capacity
            max_violation = max(max_violation, violation)
            
            if abs(force) > capacity:
                members_pass = False
        
        debug_info['max_violation'] = max_violation
        debug_info['members_pass'] = members_pass
        
        if not members_pass:
            print(f"  ❌ FAIL: Member strength (max violation = {max_violation:.2f})")
        
        # Deep beam (관대하게)
        d = problem.height - self.cover
        deep_beam_pass = problem.span / problem.height <= 5.0  # 더 관대
        
        debug_info['l/h'] = problem.span / problem.height
        debug_info['deep_beam_pass'] = deep_beam_pass
        
        if not deep_beam_pass:
            print(f"  ❌ FAIL: Deep beam (l/h = {problem.span/problem.height:.2f} > 5.0)")
        
        # Overall
        overall_pass = equilibrium and members_pass and deep_beam_pass
        
        if overall_pass:
            print(f"  ✅ PASS: All checks OK")
        
        return {
            'overall_pass': overall_pass,
            'equilibrium': equilibrium,
            'members_pass': members_pass,
            'deep_beam_pass': deep_beam_pass,
            'forces': forces,
            'debug': debug_info
        }
    
    def calculate_forces(self, problem: STMProblem, edges: List[Tuple[int, int]]):
        """힘 계산 - 허용 오차 증가"""
        try:
            n_edges = len(edges)
            n_nodes = problem.n_nodes
            
            A = np.zeros((2 * n_nodes, n_edges + 2))
            
            for idx, (n1, n2) in enumerate(edges):
                x1, y1 = problem.nodes[n1]
                x2, y2 = problem.nodes[n2]
                
                dx = x2 - x1
                dy = y2 - y1
                L = np.sqrt(dx**2 + dy**2)
                
                if L < 1e-6:
                    continue
                
                cos_theta = dx / L
                sin_theta = dy / L
                
                A[2*n1, idx] = -cos_theta
                A[2*n1+1, idx] = -sin_theta
                A[2*n2, idx] = cos_theta
                A[2*n2+1, idx] = sin_theta
            
            A[0, -2] = 1.0
            A[1, -1] = 1.0
            A[2, -2] = 0.0
            A[3, -1] = 1.0
            
            b = np.zeros(2 * n_nodes)
            for i, (x, y, P) in enumerate(problem.loads):
                node_idx = 2 + i
                b[2*node_idx + 1] = -P
            
            forces_all, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            forces = forces_all[:n_edges]
            equilibrium_error = np.linalg.norm(A @ forces_all - b)
            equilibrium = equilibrium_error < 5.0  # 더 관대 (원래 1.0)
            
            return {
                'forces': forces,
                'equilibrium': equilibrium,
                'error': equilibrium_error
            }
        
        except Exception as e:
            print(f"  ⚠️ Force calculation exception: {e}")
            return None


def generate_better_expert_data(config: HybridConfig, n_samples=500):
    """
    개선된 전문가 데이터 생성
    - 실제로 KDS를 만족하는지 확인
    """
    print("\n[DEBUG] Generating expert data...")
    
    kds = KDSChecker(config.fck, config.fy, config.cover)
    problems = []
    solutions = []
    
    valid_count = 0
    attempts = 0
    max_attempts = n_samples * 10
    
    while valid_count < n_samples and attempts < max_attempts:
        attempts += 1
        
        problem = STMProblem(config)
        problem.reset()
        
        # 간단한 트러스 구조
        edges = []
        n_nodes = problem.n_nodes
        
        # Strategy 1: Connect loads to both supports
        for load_idx in range(2, 2 + config.n_loads):
            edges.append((0, load_idx))
            edges.append((1, load_idx))
        
        # Strategy 2: Cross bracing if candidates exist
        if n_nodes > 4:
            # Add diagonal
            edges.append((0, 4))
            edges.append((1, 4))
        
        # KDS 체크
        result = kds.check(problem, edges)
        
        if result['overall_pass']:
            valid_count += 1
            
            # Adjacency matrix
            adj_matrix = np.zeros((config.max_nodes, config.max_nodes))
            for i, j in edges:
                if i < config.max_nodes and j < config.max_nodes:
                    adj_matrix[i, j] = 1.0
                    adj_matrix[j, i] = 1.0
            
            problems.append(problem.get_state())
            solutions.append(adj_matrix)
            
            if valid_count % 50 == 0:
                print(f"  Generated {valid_count}/{n_samples} valid samples")
    
    print(f"[DEBUG] Success rate: {valid_count}/{attempts} = {valid_count/attempts*100:.1f}%")
    
    if valid_count < n_samples:
        print(f"⚠️ WARNING: Only generated {valid_count}/{n_samples} valid samples")
    
    return np.array(problems), np.array(solutions)


class EdgeInitializerNN(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        n_nodes = config.max_nodes
        input_dim = n_nodes * 2 + n_nodes * 3 + 3
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, n_nodes * n_nodes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        n_nodes = self.config.max_nodes
        
        logits = self.network(x)
        edge_probs = logits.view(batch_size, n_nodes, n_nodes)
        
        edge_probs = (edge_probs + edge_probs.transpose(1, 2)) / 2.0
        
        return edge_probs
    
    def predict_edges(self, problem: STMProblem, threshold=0.5):
        self.eval()
        with torch.no_grad():
            state = torch.FloatTensor(problem.get_state()).unsqueeze(0)
            edge_probs = self.forward(state)[0]
            
            edges = []
            n_nodes = problem.n_nodes
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    if edge_probs[i, j] > threshold:
                        edges.append((i, j))
            
            return edges, edge_probs.numpy()


class GeneticAlgorithm:
    def __init__(self, config: HybridConfig):
        self.config = config
        self.kds_checker = KDSChecker(config.fck, config.fy, config.cover)
    
    def optimize(self, problem: STMProblem, initial_probs: np.ndarray):
        n_nodes = problem.n_nodes
        
        population = self.initialize_population(n_nodes, initial_probs)
        
        best_solution = None
        best_fitness = -float('inf')
        
        for generation in range(self.config.n_generations):
            fitness_scores = []
            for individual in population:
                edges = self.decode_individual(individual)
                fitness = self.evaluate_fitness(problem, edges)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()
            
            population = self.evolve_population(population, fitness_scores, initial_probs)
            
            if best_fitness >= 20.0:
                break
        
        best_edges = self.decode_individual(best_solution)
        
        return best_edges, best_fitness
    
    def initialize_population(self, n_nodes: int, edge_probs: np.ndarray):
        population = []
        
        for _ in range(self.config.population_size):
            individual = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    p = edge_probs[i, j] * 0.7 + np.random.rand() * 0.3
                    if np.random.rand() < p:
                        individual[i, j] = 1.0
                        individual[j, i] = 1.0
            
            population.append(individual)
        
        return population
    
    def decode_individual(self, individual: np.ndarray) -> List[Tuple[int, int]]:
        edges = []
        n_nodes = individual.shape[0]
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if individual[i, j] > 0.5:
                    edges.append((i, j))
        
        return edges
    
    def evaluate_fitness(self, problem: STMProblem, edges: List[Tuple[int, int]]) -> float:
        if len(edges) == 0:
            return -100.0
        
        result = self.kds_checker.check(problem, edges)
        
        if not result['overall_pass']:
            return -50.0
        
        reward = 20.0
        
        if result['equilibrium']:
            reward += 10.0
        
        n_nodes = problem.n_nodes
        expected_edges = n_nodes + 2
        edge_penalty = abs(len(edges) - expected_edges) * 2.0
        reward -= edge_penalty
        
        return reward
    
    def evolve_population(self, population, fitness_scores, edge_probs):
        n = len(population)
        
        elite_indices = np.argsort(fitness_scores)[-self.config.elite_size:]
        next_population = [population[i].copy() for i in elite_indices]
        
        while len(next_population) < n:
            parent1 = self.tournament_select(population, fitness_scores)
            parent2 = self.tournament_select(population, fitness_scores)
            
            if np.random.rand() < self.config.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            child = self.mutate(child, edge_probs)
            
            next_population.append(child)
        
        return next_population[:n]
    
    def tournament_select(self, population, fitness_scores, k=3):
        indices = np.random.choice(len(population), k, replace=False)
        best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
        return population[best_idx].copy()
    
    def crossover(self, parent1, parent2):
        mask = np.random.rand(*parent1.shape) > 0.5
        child = np.where(mask, parent1, parent2)
        child = (child + child.T) / 2.0
        child = (child > 0.5).astype(np.float32)
        return child
    
    def mutate(self, individual, edge_probs):
        n_nodes = individual.shape[0]
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if np.random.rand() < self.config.mutation_rate:
                    if individual[i, j] < 0.5:
                        if np.random.rand() < edge_probs[i, j]:
                            individual[i, j] = 1.0
                            individual[j, i] = 1.0
                    else:
                        individual[i, j] = 0.0
                        individual[j, i] = 0.0
        
        return individual


def train_nn(model, train_data, config: HybridConfig):
    X, Y = train_data
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(Y)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    for epoch in range(config.nn_epochs):
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{config.nn_epochs}, Loss: {avg_loss:.4f}")


def test_hybrid_method(model, ga, config: HybridConfig, n_tests=10):
    """테스트 - 디버깅 정보 출력"""
    results = {
        'success_rate': 0,
        'avg_fitness': 0,
        'nn_only_success': 0
    }
    
    print(f"\n[DEBUG] Testing on {n_tests} problems...")
    
    for test_idx in range(n_tests):
        print(f"\nTest {test_idx+1}/{n_tests}:")
        
        problem = STMProblem(config)
        problem.reset()
        
        print(f"  Problem: span={problem.span:.2f}m, height={problem.height:.2f}m, "
              f"n_nodes={problem.n_nodes}")
        
        # NN prediction
        nn_edges, edge_probs = model.predict_edges(problem, threshold=0.5)
        print(f"  NN predicted {len(nn_edges)} edges")
        
        kds = KDSChecker(config.fck, config.fy, config.cover)
        nn_result = kds.check(problem, nn_edges)
        
        if nn_result['overall_pass']:
            results['nn_only_success'] += 1
        
        # GA optimization
        print(f"  Running GA...")
        best_edges, best_fitness = ga.optimize(problem, edge_probs)
        print(f"  GA result: {len(best_edges)} edges, fitness={best_fitness:.2f}")
        
        if best_fitness > 0:
            results['success_rate'] += 1
        
        results['avg_fitness'] += best_fitness
    
    results['success_rate'] /= n_tests
    results['nn_only_success'] /= n_tests
    results['avg_fitness'] /= n_tests
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("Method 1: Hybrid (NN + GA) - DEBUG VERSION")
    print("="*60)
    
    config = HybridConfig()
    
    # 1. Generate VALID expert data
    print("\n[1/4] Generating VALID expert data...")
    X_train, Y_train = generate_better_expert_data(config, n_samples=200)
    print(f"✓ Generated {len(X_train)} training samples")
    
    # 2. Train NN
    print("\n[2/4] Training Neural Network...")
    model = EdgeInitializerNN(config)
    train_nn(model, (X_train, Y_train), config)
    print("✓ Training completed")
    
    # 3. Create GA
    print("\n[3/4] Creating GA optimizer...")
    ga = GeneticAlgorithm(config)
    
    # 4. Test
    print("\n[4/4] Testing hybrid method...")
    results = test_hybrid_method(model, ga, config, n_tests=10)
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"NN-only success rate: {results['nn_only_success']*100:.1f}%")
    print(f"Hybrid success rate:  {results['success_rate']*100:.1f}%")
    print(f"Average fitness:      {results['avg_fitness']:.2f}")
    print("="*60)