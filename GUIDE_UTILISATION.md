# Guide d'Utilisation - Moteur de Pricing des Dérivés

Ce guide vous accompagne pas à pas dans l'utilisation du moteur de pricing des dérivés pour des cas d'usage concrets.

## 🚀 Démarrage Rapide

### 1. Installation et Configuration

```bash
# Cloner le projet
git clone <repository-url>
cd "Derivatives pricing engine"

# Installer les dépendances
pip install numpy scipy pandas matplotlib

# Tester l'installation
python examples/basic_pricing.py
```

### 2. Premier Prix d'Option

```python
from derivatives_engine import MarketData, BlackScholesModel

# Configuration du marché
market = MarketData(
    S0=100.0,     # Prix actuel de l'actif
    K=105.0,      # Prix d'exercice
    T=0.25,       # 3 mois d'échéance  
    r=0.05,       # Taux sans risque 5%
    q=0.02,       # Rendement dividende 2%
    sigma=0.20    # Volatilité 20%
)

# Calcul du prix
prix_call = BlackScholesModel.price(market, "call")
prix_put = BlackScholesModel.price(market, "put")

print(f"Prix du call: {prix_call:.4f}€")
print(f"Prix du put: {prix_put:.4f}€")
```

## 📊 Cas d'Usage Pratiques

### Cas 1: Analyse de Sensibilité (Grecs)

```python
from derivatives_engine import MarketData, BlackScholesModel

market = MarketData(S0=100, K=100, T=0.25, r=0.05, q=0.02, sigma=0.20)

# Calcul des grecs
grecs = BlackScholesModel.greeks(market, "call")

print("=== ANALYSE DES GRECS ===")
print(f"Delta (Δ): {grecs['delta']:.4f}")
print(f"  → Pour 1€ hausse du sous-jacent: +{grecs['delta']:.4f}€")
print(f"Gamma (Γ): {grecs['gamma']:.4f}")  
print(f"  → Accélération du delta: {grecs['gamma']:.4f}")
print(f"Theta (Θ): {grecs['theta']:.4f}")
print(f"  → Érosion temporelle par jour: {grecs['theta']:.4f}€")
print(f"Vega (ν): {grecs['vega']:.4f}")
print(f"  → Sensibilité volatilité (+1%): +{grecs['vega']:.4f}€")
print(f"Rho (ρ): {grecs['rho']:.4f}")
print(f"  → Sensibilité taux (+1%): +{grecs['rho']:.4f}€")
```

### Cas 2: Options Exotiques (Barrières)

```python
from derivatives_engine import ExoticOptions, MarketData

market = MarketData(S0=100, K=100, T=0.25, r=0.05, q=0.02, sigma=0.20)
exotic = ExoticOptions()

# Option barrière up-and-out
prix, ic = exotic.barrier_option_mc(
    market_data=market,
    option_type="call",
    barrier_type="up_out",
    barrier_level=120.0,
    n_paths=100000,
    n_steps=252
)

print(f"Option barrière: {prix:.4f}€ ± {ic:.4f}€")

# Comparaison avec option vanilla
prix_vanilla = BlackScholesModel.price(market, "call")
rabais = prix_vanilla - prix

print(f"Option vanilla: {prix_vanilla:.4f}€")
print(f"Rabais barrière: {rabais:.4f}€ ({rabais/prix_vanilla*100:.1f}%)")
```

### Cas 3: Modèle Heston (Volatilité Stochastique)

```python
from derivatives_engine import HestonModel, HestonParameters, MarketData

# Paramètres du modèle Heston
params_heston = HestonParameters(
    v0=0.04,       # Variance initiale (vol 20%)
    theta=0.04,    # Variance long terme  
    kappa=2.0,     # Vitesse retour moyenne
    sigma_v=0.3,   # Vol de la volatilité
    rho=-0.7       # Corrélation (effet de levier)
)

heston = HestonModel(params_heston)
market = MarketData(S0=100, K=100, T=0.25, r=0.05, q=0.02, sigma=0.20)

# Pricing Monte Carlo
prix_heston, ic = heston.monte_carlo_price(market, "call", n_paths=50000)

print(f"Prix Heston: {prix_heston:.4f}€ ± {ic:.4f}€")

# Comparaison Black-Scholes
prix_bs = BlackScholesModel.price(market, "call")
difference = prix_heston - prix_bs

print(f"Prix Black-Scholes: {prix_bs:.4f}€")
print(f"Impact vol stochastique: {difference:+.4f}€")
```

### Cas 4: Calibration de Modèle

```python
from derivatives_engine.calibration import HestonCalibrator
import pandas as pd

# Données de marché (prix d'options observés)
donnees_marche = pd.DataFrame({
    'strike': [90, 95, 100, 105, 110],
    'bid': [12.5, 8.2, 4.5, 2.1, 0.8],
    'ask': [13.0, 8.7, 5.0, 2.6, 1.3],
    'option_type': ['call'] * 5
})

market = MarketData(S0=100, K=100, T=0.25, r=0.05, q=0.02, sigma=0.20)

# Calibration automatique
calibrateur = HestonCalibrator()
params_calibres, metriques = calibrateur.calibrate_to_market_data(
    market, donnees_marche
)

print("=== RÉSULTATS CALIBRATION ===")
print(f"Erreur RMSE: {metriques['rmse']:.4f}")
print(f"v0 calibré: {params_calibres.v0:.4f}")
print(f"theta calibré: {params_calibres.theta:.4f}")
print(f"kappa calibré: {params_calibres.kappa:.4f}")
```

### Cas 5: Portfolio et Gestion des Risques

```python
from derivatives_engine import DerivativesPricingEngine
from derivatives_engine.utils import RiskAnalyzer

# Définir un portefeuille
positions = [
    {
        'market_data': MarketData(S0=100, K=95, T=0.25, r=0.05, sigma=0.20),
        'option_type': 'call',
        'quantity': 100,  # Long 100 calls
        'description': 'Call protecteur'
    },
    {
        'market_data': MarketData(S0=100, K=105, T=0.25, r=0.05, sigma=0.20),
        'option_type': 'call', 
        'quantity': -50,  # Short 50 calls
        'description': 'Call couvert'
    },
    {
        'market_data': MarketData(S0=100, K=95, T=0.25, r=0.05, sigma=0.20),
        'option_type': 'put',
        'quantity': -75,  # Short 75 puts
        'description': 'Put vendu'
    }
]

engine = DerivativesPricingEngine()
risk_analyzer = RiskAnalyzer()

# Analyse des risques
print("=== ANALYSE DU PORTEFEUILLE ===")

for i, pos in enumerate(positions, 1):
    prix = BlackScholesModel.price(pos['market_data'], pos['option_type'])
    valeur = prix * pos['quantity']
    print(f"{i}. {pos['description']}: {valeur:+.2f}€")

# VaR du portefeuille
var_result = risk_analyzer.calculate_var(positions, confidence_level=0.95)
print(f"\nVaR 95%: {var_result.var:.2f}€")
print(f"Expected Shortfall: {var_result.expected_shortfall:.2f}€")
```

## 🔧 Configuration et Optimisation

### Configuration Globale

```python
from derivatives_engine.utils import get_config, ConfigManager

# Accéder à la configuration
config = get_config()

print(f"Chemins Monte Carlo par défaut: {config.numerical.default_monte_carlo_paths}")
print(f"JIT activé: {config.performance.enable_jit}")

# Modifier la configuration
config_manager = ConfigManager()
config_manager.update_config({
    'numerical': {
        'default_monte_carlo_paths': 200000,
        'random_seed': 42
    },
    'performance': {
        'enable_jit': True,
        'max_threads': 4
    }
})
```

### Optimisations Performance

```python
from derivatives_engine.utils import OptimizedPricingEngine, BenchmarkSuite

# Moteur optimisé
engine_opt = OptimizedPricingEngine()

# Pricing vectorisé pour plusieurs options
options_list = [
    MarketData(S0=100, K=k, T=0.25, r=0.05, sigma=0.20) 
    for k in [95, 100, 105, 110, 115]
]

# Calcul en parallèle
prix_vect = engine_opt.vectorized_pricing(options_list, "call")
print("Prix vectorisés:", prix_vect)

# Benchmark de performance
benchmark = BenchmarkSuite()
resultats = benchmark.comprehensive_benchmark()
print("Temps de calcul par modèle:")
for modele, temps in resultats.items():
    print(f"  {modele}: {temps:.2f}ms")
```

### Logging et Debugging

```python
from derivatives_engine.utils import PricingEngineLogger

# Activer le mode debug
PricingEngineLogger.enable_debug_mode()

# Calculs avec logs détaillés
market = MarketData(S0=100, K=100, T=0.25, r=0.05, sigma=0.20)
prix = BlackScholesModel.price(market, "call")

# Les logs seront dans le répertoire logs/
# - pricing_engine.log : logs généraux
# - errors.log : erreurs uniquement
```

## 📈 Analyses Avancées

### Surface de Volatilité

```python
import numpy as np
import matplotlib.pyplot as plt

# Grille de strikes et maturités
strikes = np.arange(80, 121, 5)
maturites = np.arange(0.1, 1.1, 0.1)

# Matrice des prix
prix_matrix = np.zeros((len(maturites), len(strikes)))

for i, T in enumerate(maturites):
    for j, K in enumerate(strikes):
        market = MarketData(S0=100, K=K, T=T, r=0.05, sigma=0.20)
        prix_matrix[i, j] = BlackScholesModel.price(market, "call")

# Visualisation
plt.figure(figsize=(12, 8))
plt.contourf(strikes, maturites, prix_matrix, levels=20, cmap='viridis')
plt.colorbar(label='Prix de l\'option')
plt.xlabel('Strike')
plt.ylabel('Maturité (années)')
plt.title('Surface des Prix d\'Options Call')
plt.show()
```

### Test de Stress

```python
# Scénarios de stress
scenarios = {
    'crash_marche': {'S0': 0.8, 'sigma': 1.5},      # -20% prix, +50% vol
    'crise_liquidite': {'r': 2.0, 'sigma': 2.0},    # Taux x2, vol x2  
    'volatilite_extreme': {'sigma': 3.0},            # Vol x3
    'taux_negatifs': {'r': -0.01}                    # Taux -1%
}

market_base = MarketData(S0=100, K=100, T=0.25, r=0.05, sigma=0.20)
prix_base = BlackScholesModel.price(market_base, "call")

print("=== TEST DE STRESS ===")
print(f"Prix de base: {prix_base:.4f}€")

for nom_scenario, chocs in scenarios.items():
    # Appliquer les chocs
    market_choque = market_base.copy(**chocs)
    prix_choque = BlackScholesModel.price(market_choque, "call")
    
    impact = prix_choque - prix_base
    impact_pct = impact / prix_base * 100
    
    print(f"{nom_scenario}: {prix_choque:.4f}€ ({impact:+.4f}€, {impact_pct:+.1f}%)")
```

## 🐛 Gestion des Erreurs

### Validation des Entrées

```python
from derivatives_engine.utils import ValidationError, ModelError

# Exemples d'erreurs communes et leur gestion
try:
    # Prix négatif (erreur)
    market_invalide = MarketData(S0=-50, K=100, T=0.25, r=0.05, sigma=0.20)
except ValidationError as e:
    print(f"Erreur validation: {e}")

try:
    # Volatilité négative (erreur)
    market_invalide = MarketData(S0=100, K=100, T=0.25, r=0.05, sigma=-0.20)
except ValidationError as e:
    print(f"Erreur validation: {e}")

try:
    # Maturité négative (erreur)
    market_invalide = MarketData(S0=100, K=100, T=-1, r=0.05, sigma=0.20)
except ValidationError as e:
    print(f"Erreur validation: {e}")
```

### Problèmes de Convergence

```python
from derivatives_engine import HestonModel, HestonParameters
from derivatives_engine.utils import ConvergenceError

# Paramètres problématiques (violation condition de Feller)
params_problemes = HestonParameters(
    v0=0.01, theta=0.01, kappa=0.5, sigma_v=1.0, rho=0.9
)

try:
    heston = HestonModel(params_problemes)
    # Un warning sera émis mais le modèle sera créé
except Exception as e:
    print(f"Attention: {e}")

# Pour des calculs robustes, augmenter le nombre de chemins
market = MarketData(S0=100, K=100, T=0.25, r=0.05, sigma=0.20)
prix, ic = heston.monte_carlo_price(market, "call", n_paths=200000)
print(f"Prix robuste: {prix:.4f}€ ± {ic:.4f}€")
```

## 🎯 Conseils d'Utilisation

### 1. Choix du Modèle

**Black-Scholes** : 
- ✅ Rapide, analytique, ideal pour options vanilla
- ❌ Suppose volatilité constante

**Heston** :
- ✅ Volatilité stochastique réaliste, smile de volatilité
- ❌ Plus lent, paramètres à calibrer

**Jump-Diffusion** :
- ✅ Capture les événements extrêmes
- ❌ Complexité additionnelle

### 2. Paramètres Monte Carlo

```python
# Règles générales pour Monte Carlo
situations = {
    'tests_rapides': {'n_paths': 10000, 'n_steps': 50},
    'calculs_standard': {'n_paths': 100000, 'n_steps': 252}, 
    'precision_elevee': {'n_paths': 500000, 'n_steps': 252},
    'options_complexes': {'n_paths': 1000000, 'n_steps': 1000}
}
```

### 3. Performance

```python
# Optimisations recommandées
import time

def benchmark_pricing(market, option_type, n_iterations=1000):
    start = time.time()
    
    for _ in range(n_iterations):
        prix = BlackScholesModel.price(market, option_type)
    
    temps_total = time.time() - start
    temps_moyen = temps_total / n_iterations * 1000
    
    print(f"Temps moyen: {temps_moyen:.3f}ms par option")
    print(f"Débit: {n_iterations/temps_total:.0f} options/seconde")

market = MarketData(S0=100, K=100, T=0.25, r=0.05, sigma=0.20)
benchmark_pricing(market, "call")
```

## 🔗 Ressources Complémentaires

- **Documentation API** : Voir docstrings dans le code
- **Tests unitaires** : Dossier `test/` pour exemples d'usage
- **Exemples avancés** : Dossier `examples/`
- **Configuration** : Fichier `config/pricing_engine.json`

---

*Pour des questions spécifiques, consultez la documentation technique ou créez une issue GitHub.*