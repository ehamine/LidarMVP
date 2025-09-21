#!/bin/bash

# Script d'automatisation pour tests du simulateur Ouster PCAP
# Usage: ./test_automation.sh [test_type]

set -e

# Configuration
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../build" && pwd)"
SIMULATOR="$BUILD_DIR/ouster_sim"
MOCK_GEN="$BUILD_DIR/examples/mock_pcap_generator"
UDP_LISTENER="$BUILD_DIR/examples/udp_listener"
TEST_DIR="/tmp/ouster_sim_tests"
LOG_FILE="$TEST_DIR/test_results.log"

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonctions utilitaires
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}✗${NC} $1" | tee -a "$LOG_FILE"
}

# Initialisation
init_test_env() {
    # Créer répertoire de test d'abord
    mkdir -p "$TEST_DIR"

    log "Initialisation de l'environnement de test"
    cd "$TEST_DIR"

    # Vérifier exécutables
    if [[ ! -x "$SIMULATOR" ]]; then
        error "Simulateur non trouvé: $SIMULATOR"
        exit 1
    fi

    if [[ ! -x "$MOCK_GEN" ]]; then
        error "Générateur PCAP non trouvé: $MOCK_GEN"
        exit 1
    fi

    success "Environnement initialisé"
}

# Test 1: Fonctionnalité de base
test_basic_functionality() {
    log "=== Test 1: Fonctionnalité de base ==="

    # Générer PCAP test
    log "Génération PCAP de test..."
    "$MOCK_GEN" basic_test.pcap 100 10

    # Test simulation basique
    log "Test simulation basique..."
    timeout 30 "$SIMULATOR" --pcap basic_test.pcap --verbose > basic_test.log 2>&1

    # Vérifier résultats
    if grep -q "Simulation completed" basic_test.log; then
        success "Simulation basique réussie"
    else
        error "Échec simulation basique"
        return 1
    fi

    # Vérifier classification
    if grep -q "100 LIDAR, 10 IMU" basic_test.log; then
        success "Classification correcte"
    else
        warning "Classification inattendue"
    fi
}

# Test 2: Performance et stress
test_performance() {
    log "=== Test 2: Performance et stress ==="

    # Générer gros PCAP
    log "Génération PCAP de performance..."
    "$MOCK_GEN" perf_test.pcap 5000 500

    # Test vitesse élevée
    log "Test vitesse 5x..."
    timeout 60 "$SIMULATOR" --pcap perf_test.pcap --rate 5.0 --verbose > perf_test.log 2>&1

    # Analyser performance
    local pps=$(grep "packets_per_second" perf_test.log | grep -o '[0-9.]*' | tail -1)
    if (( $(echo "$pps > 1000" | bc -l) )); then
        success "Performance acceptable: ${pps} pps"
    else
        warning "Performance faible: ${pps} pps"
    fi
}

# Test 3: Options de timing
test_timing_options() {
    log "=== Test 3: Options de timing ==="

    # Générer PCAP pour timing
    "$MOCK_GEN" timing_test.pcap 50 5

    # Test avec jitter
    log "Test avec jitter..."
    timeout 30 "$SIMULATOR" --pcap timing_test.pcap --jitter 0.001 --verbose > jitter_test.log 2>&1

    # Test sans timestamps
    log "Test mode timing fixe..."
    timeout 30 "$SIMULATOR" --pcap timing_test.pcap --no-timestamps --verbose > notimestamp_test.log 2>&1

    # Test boucle (courte)
    log "Test mode boucle..."
    timeout 10 "$SIMULATOR" --pcap timing_test.pcap --loop --rate 10.0 --verbose > loop_test.log 2>&1

    if grep -q "Simulation completed" jitter_test.log && \
       grep -q "Simulation completed" notimestamp_test.log; then
        success "Options de timing fonctionnelles"
    else
        error "Problème avec options de timing"
        return 1
    fi
}

# Test 4: Network et UDP
test_network() {
    log "=== Test 4: Test réseau et UDP ==="

    # Démarrer listeners en background
    log "Démarrage listeners UDP..."
    "$UDP_LISTENER" 8502 > lidar_listener.log 2>&1 &
    local lidar_pid=$!
    "$UDP_LISTENER" 8503 > imu_listener.log 2>&1 &
    local imu_pid=$!

    # Attendre démarrage
    sleep 2

    # Test avec destinations personnalisées
    log "Test transmission UDP..."
    timeout 30 "$SIMULATOR" --pcap basic_test.pcap \
        --dst-lidar 127.0.0.1:8502 \
        --dst-imu 127.0.0.1:8503 \
        --verbose > network_test.log 2>&1

    # Arrêter listeners
    kill $lidar_pid $imu_pid 2>/dev/null || true
    sleep 1

    # Vérifier réception
    if grep -q "Packets:" lidar_listener.log && grep -q "Packets:" imu_listener.log; then
        success "Transmission UDP réussie"
    else
        error "Problème transmission UDP"
        return 1
    fi
}

# Test 5: Gestion d'erreurs
test_error_handling() {
    log "=== Test 5: Gestion d'erreurs ==="

    # Test fichier inexistant
    log "Test fichier PCAP inexistant..."
    if "$SIMULATOR" --pcap inexistant.pcap 2>/dev/null; then
        error "Devrait échouer avec fichier inexistant"
        return 1
    else
        success "Erreur fichier inexistant gérée"
    fi

    # Test arguments invalides
    log "Test arguments invalides..."
    if "$SIMULATOR" --rate invalid 2>/dev/null; then
        error "Devrait échouer avec rate invalide"
        return 1
    else
        success "Erreur arguments invalides gérée"
    fi
}

# Test 6: Métriques et logging
test_metrics() {
    log "=== Test 6: Métriques et logging ==="

    # Test logging fichier
    log "Test logging vers fichier..."
    timeout 30 "$SIMULATOR" --pcap basic_test.pcap --log metrics_test.log --verbose > /dev/null 2>&1

    if [[ -f "metrics_test.log" ]] && [[ -s "metrics_test.log" ]]; then
        success "Logging fichier fonctionnel"
    else
        warning "Problème logging fichier"
    fi

    # Test métriques HTTP (port libre)
    log "Test métriques HTTP..."
    timeout 10 "$SIMULATOR" --pcap basic_test.pcap --metrics-port 9999 --verbose > metrics_http.log 2>&1 &
    local sim_pid=$!

    sleep 3
    if curl -s http://localhost:9999/metrics > /dev/null 2>&1; then
        success "Métriques HTTP fonctionnelles"
    else
        warning "Métriques HTTP non accessibles"
    fi

    kill $sim_pid 2>/dev/null || true
}

# Rapport de résultats
generate_report() {
    log "=== Rapport de test final ==="

    local total_tests=6
    local passed_tests=0

    # Compter succès
    passed_tests=$(grep -c "✓" "$LOG_FILE" 2>/dev/null || echo 0)
    local warnings=$(grep -c "⚠" "$LOG_FILE" 2>/dev/null || echo 0)
    local errors=$(grep -c "✗" "$LOG_FILE" 2>/dev/null || echo 0)

    echo
    log "RÉSULTATS:"
    log "  Tests réussis: ${passed_tests}"
    log "  Avertissements: ${warnings}"
    log "  Erreurs: ${errors}"

    # Statistiques fichiers
    log "FICHIERS GÉNÉRÉS:"
    ls -la *.pcap *.log 2>/dev/null | while read line; do
        log "  $line"
    done

    if [[ $errors -eq 0 ]]; then
        success "TOUS LES TESTS PRINCIPAUX RÉUSSIS"
        return 0
    else
        error "CERTAINS TESTS ONT ÉCHOUÉ"
        return 1
    fi
}

# Nettoyage
cleanup() {
    log "Nettoyage..."

    # Tuer processus résiduels
    pkill -f "udp_listener" 2>/dev/null || true
    pkill -f "ouster_sim" 2>/dev/null || true

    # Garder logs pour analyse
    log "Logs sauvegardés dans: $TEST_DIR"
    log "Fichier de log principal: $LOG_FILE"
}

# Menu principal
show_usage() {
    echo "Usage: $0 [test_type]"
    echo
    echo "Types de test disponibles:"
    echo "  all         - Tous les tests (défaut)"
    echo "  basic       - Test fonctionnalité de base"
    echo "  performance - Test de performance"
    echo "  timing      - Test options de timing"
    echo "  network     - Test réseau et UDP"
    echo "  errors      - Test gestion d'erreurs"
    echo "  metrics     - Test métriques et logging"
    echo
    echo "Exemples:"
    echo "  $0                # Tous les tests"
    echo "  $0 basic          # Test de base seulement"
    echo "  $0 performance    # Test de performance seulement"
}

# Point d'entrée principal
main() {
    local test_type=${1:-all}

    case $test_type in
        -h|--help)
            show_usage
            exit 0
            ;;
        basic)
            init_test_env
            test_basic_functionality
            ;;
        performance)
            init_test_env
            test_performance
            ;;
        timing)
            init_test_env
            test_timing_options
            ;;
        network)
            init_test_env
            test_network
            ;;
        errors)
            init_test_env
            test_error_handling
            ;;
        metrics)
            init_test_env
            test_metrics
            ;;
        all)
            log "DÉBUT DES TESTS COMPLETS"
            init_test_env
            test_basic_functionality || true
            test_performance || true
            test_timing_options || true
            test_network || true
            test_error_handling || true
            test_metrics || true
            generate_report
            ;;
        *)
            echo "Type de test inconnu: $test_type"
            show_usage
            exit 1
            ;;
    esac

    cleanup
}

# Gestion des signaux
trap cleanup EXIT INT TERM

# Exécution
main "$@"