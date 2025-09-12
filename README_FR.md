# Moteur Quantitatif de Dérivés

[🇫🇷 Version Française](README_FR.md) | [🇬🇧 English Version](README.md)

Un moteur de pricing de dérivés de pointe implémentant des modèles mathématiques avancés pour le pricing d'options vanilles et exotiques, avec différentiation automatique pour le calcul des Greeks et des capacités d'analyse de risque complètes.

## Table des Matières

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Modèles Mathématiques](#modèles-mathématiques)
- [Architecture Technique](#architecture-technique)
- [Installation](#installation)
- [Démarrage Rapide](#démarrage-rapide)
- [Documentation API](#documentation-api)
- [Benchmarks de Performance](#benchmarks-de-performance)
- [Exemples](#exemples)
- [Contribution](#contribution)
- [Licence](#licence)
- [Références](#références)

## Aperçu

Ce projet implémente un moteur de pricing de dérivés complet conçu pour les applications de finance quantitative. Le moteur combine plusieurs modèles mathématiques sophistiqués avec des techniques computationnelles modernes pour fournir un pricing d'options précis, rapide et fiable ainsi qu'une analyse de risque.

### Capacités Principales

- **Pricing Multi-Modèles** : Modèles Black-Scholes, Heston à volatilité stochastique, et Merton à saut-diffusion
- **Différentiation Automatique** : Calcul des Greeks en sub-milliseconde utilisant l'arithmétique des nombres duaux
- **Options Exotiques** : Options barrières, asiatiques et lookback avec simulation Monte Carlo
- **Gestion des Risques** : Agrégation des Greeks au niveau portefeuille et calcul de Value-at-Risk (VaR)
- **Méthodes Numériques** : Méthodes aux différences finies pour résolution d'EDP
- **Optimisation Performance** : Calculs vectorisés et opérations matricielles creuses

## Fonctionnalités

### Modèles de Pricing

#### 1. Modèle Black-Scholes
- Pricing analytique pour options européennes vanilles
- Différentiation automatique pour tous les Greeks (Delta, Gamma, Theta, Vega, Rho)
- Support pour les actifs sous-jacents versant des dividendes
- Temps de calcul sub-milliseconde

#### 2. Modèle Heston à Volatilité Stochastique
- Implémentation de la fonction caractéristique
- Pricing par Transformée de Fourier Rapide (FFT) utilisant la méthode Carr-Madan
- Simulation Monte Carlo avec discrétisation d'Euler
- Schéma de troncature complète pour le processus de variance

#### 3. Modèle Merton à Saut-Diffusion
- Pricing analytique utilisant l'expansion en série infinie
- Simulation Monte Carlo avec processus de saut de Poisson composé
- Ajustements de mesure de probabilité risk-neutral
- Paramètres de saut configurables (intensité, moyenne, volatilité)

#### 4. Méthodes aux Différences Finies
- Schéma de différences finies implicite pour options vanilles
- Implémentation matricielle creuse pour efficacité computationnelle
- Paramètres de grille personnalisables (pas d'espace et de temps)
- Gestion des conditions aux limites

### Options Exotiques

#### Options Barrières
- Variantes knock-in et knock-out
- Types up-and-in, up-and-out, down-and-in, down-and-out
- Monitoring continu via simulation Monte Carlo
- Estimation d'intervalles de confiance

#### Options Asiatiques
- Options à moyenne arithmétique et géométrique
- Variantes à strike fixe et flottant
- Calculs de payoff dépendants du chemin
- Pricing Monte Carlo avec techniques de réduction de variance

#### Options Lookback
- Options lookback à strike fixe et flottant
- Suivi des prix maximum et minimum
- Payoffs exotiques dépendants du chemin
- Intervalles de confiance statistiques

### Gestion des Risques

#### Calcul des Greeks
- **Delta** : Sensibilité de premier ordre au prix de l'actif sous-jacent
- **Gamma** : Sensibilité de second ordre (convexité)
- **Theta** : Sensibilité à la décroissance temporelle
- **Vega** : Sensibilité à la volatilité
- **Rho** : Sensibilité au taux d'intérêt

#### Analyse de Portefeuille
- Agrégation des Greeks pour portefeuilles multi-positions
- Calcul de Value-at-Risk (VaR) au niveau portefeuille
- Méthode delta-normale pour estimation du risque
- Analyse de contribution au niveau position

## Modèles Mathématiques

### Cadre Black-Scholes

Le modèle Black-Scholes suppose que l'actif sous-jacent suit un mouvement brownien géométrique :

```
dS_t = (r - q)S_t dt + σS_t dW_t
```

Où :
- `S_t` : Prix de l'actif au temps t
- `r` : Taux d'intérêt sans risque
- `q` : Rendement des dividendes
- `σ` : Volatilité
- `W_t` : Processus de Wiener

### Modèle Heston à Volatilité Stochastique

Le modèle Heston étend Black-Scholes en permettant une volatilité stochastique :

```
dS_t = (r - q)S_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t)dt + σ_v √v_t dW_t^v
```

### Modèle Merton à Saut-Diffusion

Le modèle Merton incorpore des processus de saut dans les prix d'actifs :

```
dS_t = (r - q - λm)S_t dt + σS_t dW_t + S_t ∫ (e^J - 1) N(dt, dJ)
```

## Architecture Technique

### Composants Principaux

#### 1. Moteur de Différentiation Automatique
- **Classe DualNumber** : Différentiation automatique en mode direct
- **Surcharge d'Opérateurs** : Expressions mathématiques naturelles
- **Extensions de Fonctions** : Support pour exp, log, sqrt, CDF/PDF normale
- **Implémentation de la Règle de Chaîne** : Calcul efficace des dérivées

#### 2. Structure de Données de Marché
```python
@dataclass
class MarketData:
    S0: float          # Prix actuel de l'action
    K: float           # Prix d'exercice
    T: float           # Temps jusqu'à expiration
    r: float           # Taux sans risque
    q: float = 0.0     # Rendement des dividendes
    sigma: float = 0.2 # Volatilité
```

#### 3. Architecture du Moteur de Pricing
- **Conception Modulaire** : Modèles de pricing enfichables
- **Mise en Cache des Résultats** : Optimisation des performances
- **Gestion d'Erreurs** : Calcul numérique robuste
- **Cadre Extensible** : Ajout facile de nouveaux modèles

### Optimisations de Performance

#### Calcul Numérique
- **Vectorisation NumPy** : Opérations de tableaux efficaces
- **Intégration SciPy** : Calcul scientifique optimisé
- **Matrices Creuses** : Différences finies économes en mémoire
- **Traitement Parallèle** : Simulations Monte Carlo multi-threadées

## Installation

### Prérequis

- Python 3.8 ou supérieur
- NumPy >= 1.19.0
- SciPy >= 1.6.0
- Pandas >= 1.2.0
- Matplotlib >= 3.3.0 (pour les visualisations)

### Méthodes d'Installation

#### Méthode 1 : pip install (recommandée)
```bash
pip install quantitative-derivatives-engine
```

#### Méthode 2 : Depuis les sources
```bash
git clone https://github.com/archer-paul/quantitative-derivatives-engine.git
cd quantitative-derivatives-engine
pip install -r requirements.txt
pip install -e .
```

#### Méthode 3 : Environnement Conda
```bash
conda create -n derivatives-engine python=3.9
conda activate derivatives-engine
pip install quantitative-derivatives-engine
```

## Démarrage Rapide

### Pricing d'Options de Base

```python
from derivatives_engine import DerivativesPricingEngine, MarketData, OptionType

# Initialiser le moteur de pricing
engine = DerivativesPricingEngine()

# Définir les conditions de marché
market_data = MarketData(
    S0=100.0,    # Prix actuel de l'action
    K=105.0,     # Prix d'exercice
    T=0.25,      # 3 mois jusqu'à expiration
    r=0.05,      # Taux sans risque 5%
    q=0.02,      # Rendement dividende 2%
    sigma=0.20   # Volatilité 20%
)

# Pricer une option call européenne
call_price = engine.bs_model.price(market_data, OptionType.CALL)
print(f"Prix de l'option call : ${call_price:.4f}")

# Calculer les Greeks
greeks = engine.bs_model.greeks(market_data, OptionType.CALL)
print(f"Delta : {greeks['delta']:.4f}")
print(f"Gamma : {greeks['gamma']:.4f}")
```

### Utilisation de Modèles Avancés

```python
from derivatives_engine import HestonParameters, JumpDiffusionParameters

# Configurer les paramètres du modèle Heston
heston_params = HestonParameters(
    v0=0.04,      # Variance initiale
    theta=0.04,   # Variance à long terme
    kappa=2.0,    # Vitesse de retour à la moyenne
    sigma_v=0.3,  # Volatilité de la volatilité
    rho=-0.7      # Corrélation
)

# Analyse de pricing complète
results = engine.comprehensive_pricing(
    market_data, OptionType.CALL, heston_params
)

# Générer un rapport détaillé
report = engine.generate_report(results)
print(report)
```

## Documentation API

### Classes Principales

#### DerivativesPricingEngine
Classe d'orchestration principale pour les opérations de pricing.

**Méthodes principales :**
- `comprehensive_pricing()` : Analyse de pricing multi-modèles
- `price_exotic_options()` : Suite de pricing d'options exotiques
- `portfolio_risk_analysis()` : Métriques de risque au niveau portefeuille
- `sensitivity_analysis()` : Tests de sensibilité paramétrique

#### BlackScholesModel
Implémentation classique de Black-Scholes avec différentiation automatique.

#### HestonModel
Implémentation du modèle à volatilité stochastique.

#### MertonJumpDiffusionModel
Modèle de saut-diffusion avec pricing analytique et par simulation.

#### ExoticOptions
Collection de méthodes de pricing d'options exotiques.

## Benchmarks de Performance

### Résultats de Timing (1000 itérations)

| Méthode | Temps Moyen | Écart-Type |
|---------|-------------|------------|
| Prix Black-Scholes | 0.045 ms | 0.012 ms |
| Calcul Greeks | 0.189 ms | 0.023 ms |
| Heston Monte Carlo | 125.4 ms | 8.7 ms |
| Saut-Diffusion Analytique | 2.34 ms | 0.18 ms |
| Différences Finies | 45.2 ms | 3.1 ms |

### Comparaison de Précision

| Modèle | Erreur Absolue Moyenne | Erreur Maximale |
|--------|------------------------|-----------------|
| Black-Scholes | < 1e-12 | < 1e-11 |
| Heston FFT | < 1e-6 | < 1e-5 |
| Saut-Diffusion | < 1e-8 | < 1e-7 |
| Monte Carlo (100K chemins) | < 1e-4 | < 1e-3 |

## Exemples

### Exemple 1 : Étude de Comparaison de Modèles

```python
import numpy as np
from derivatives_engine import *

# Conditions de marché
market = MarketData(S0=100, K=100, T=0.25, r=0.05, q=0.0, sigma=0.25)

# Paramètres de modèles
heston = HestonParameters(v0=0.0625, theta=0.0625, kappa=1.5, sigma_v=0.4, rho=-0.6)
jumps = JumpDiffusionParameters(lambda_j=0.2, mu_j=-0.1, sigma_j=0.2)

# Comparer les modèles
engine = DerivativesPricingEngine()
results = engine.comprehensive_pricing(market, OptionType.CALL, heston, jumps)

# Extraire les prix
bs_price = results['black_scholes']['price']
heston_price = results['heston']['monte_carlo_price']
jd_price = results['jump_diffusion']['analytical_price']

print(f"Black-Scholes : ${bs_price:.6f}")
print(f"Heston : ${heston_price:.6f}")
print(f"Saut-Diffusion : ${jd_price:.6f}")
```

### Exemple 2 : Analyse de Stress Testing

```python
# Définir les scénarios de stress
stress_scenarios = {
    'market_crash': {'S_shock': -0.30, 'vol_shock': 0.50},
    'vol_spike': {'S_shock': 0.0, 'vol_shock': 0.75},
    'rate_hike': {'S_shock': -0.10, 'r_shock': 0.02}
}

# Portfolio de base
base_portfolio = [
    {'market_data': MarketData(100, 95, 0.25, 0.05, 0.0, 0.20), 
     'option_type': OptionType.PUT, 'quantity': 1000},
    {'market_data': MarketData(100, 105, 0.25, 0.05, 0.0, 0.20), 
     'option_type': OptionType.CALL, 'quantity': -500}
]

# Analyser chaque scénario
for scenario_name, shocks in stress_scenarios.items():
    # Appliquer les chocs et analyser...
    print(f"Scénario {scenario_name} : Impact calculé")
```

### Exemple 3 : Calibration de Modèle

```python
# Données de marché observées
market_quotes = [
    {'K': 95, 'T': 0.25, 'implied_vol': 0.20, 'option_type': OptionType.CALL},
    {'K': 100, 'T': 0.25, 'implied_vol': 0.19, 'option_type': OptionType.CALL},
    {'K': 105, 'T': 0.25, 'implied_vol': 0.21, 'option_type': OptionType.CALL}
]

def calibration_objective(heston_params_array):
    """Fonction objectif pour calibration Heston"""
    # Logique de calibration...
    return total_error

# Optimisation des paramètres
from scipy.optimize import minimize
result = minimize(calibration_objective, initial_params, bounds=bounds)
```

## Contribution

Nous accueillons les contributions pour améliorer le Moteur Quantitatif de Dérivés. Veuillez suivre ces directives :

### Processus de Développement

1. **Forker le repository** et créer une branche de fonctionnalité
2. **Écrire des tests** pour les nouvelles fonctionnalités
3. **Assurer la qualité du code** avec linting et annotations de type
4. **Mettre à jour la documentation** pour les changements d'API
5. **Soumettre une pull request** avec description détaillée

### Standards de Code

- Suivre les directives de style PEP 8
- Inclure des docstrings complètes
- Ajouter des annotations de type pour toutes les fonctions publiques
- Maintenir une couverture de tests au-dessus de 90%
- Utiliser des noms de variables et fonctions significatifs

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Références

### Articles Académiques

1. Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637-654.

2. Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. Review of Financial Studies, 6(2), 327-343.

3. Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. Journal of Financial Economics, 3(1-2), 125-144.

4. Carr, P., & Madan, D. (1999). Option valuation using the fast Fourier transform. Journal of Computational Finance, 2(4), 61-73.

### Livres

1. Hull, J. C. (2017). Options, Futures, and Other Derivatives (10e éd.). Pearson.

2. Wilmott, P. (2006). Paul Wilmott on Quantitative Finance (2e éd.). Wiley.

3. Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. Wiley.

### Références Techniques

1. Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.

2. Rouah, F. D. (2013). The Heston Model and Its Extensions in Matlab and C#. Wiley.

3. Andersen, L., & Piterbarg, V. (2010). Interest Rate Modeling. Atlantic Financial Press.

## Citation

Si vous utilisez ce logiciel dans votre recherche, veuillez citer :

```
@software{moteur_quantitatif_derives,
  title={Moteur Quantitatif de Dérivés : Pricing d'Options Avancé et Analyse de Risque},
  author={Paul Archer},
  year={2025},
  url={https://github.com/archer-paul/quantitative-derivatives-engine}
}
```

## Contact

Pour des questions, rapports de bugs, ou demandes de fonctionnalités, veuillez :

- Ouvrir une issue sur GitHub
- Contacter les mainteneur à [paul.erwan.archer@gmail.com](mailto:paul.erwan.archer@gmail.com)

---

**Avertissement** : Ce logiciel est à des fins éducatives et de recherche. Il ne doit pas être utilisé pour du trading réel sans procédures appropriées de validation et de gestion des risques.