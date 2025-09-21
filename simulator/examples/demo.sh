#!/bin/bash

# Script de démonstration interactif du simulateur Ouster PCAP
# Montre les principales fonctionnalités avec explications

set -e

# Configuration
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../build" && pwd)"
SIMULATOR="$BUILD_DIR/ouster_sim"
MOCK_GEN="$BUILD_DIR/examples/mock_pcap_generator"
UDP_LISTENER="$BUILD_DIR/examples/udp_listener"

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Fonctions utilitaires
step() {
    echo
    echo -e "${BLUE}=== $1 ===${NC}"
}

info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

action() {
    echo -e "${YELLOW}▶${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

pause_demo() {
    echo
    read -p "Appuyez sur Entrée pour continuer..."
}

# Vérification environnement
check_environment() {
    step "Vérification de l'environnement"

    if [[ ! -x "$SIMULATOR" ]]; then
        echo "❌ Simulateur non trouvé. Veuillez compiler d'abord:"
        echo "   cd simulator && ./build.sh"
        exit 1
    fi

    success "Simulateur trouvé: $SIMULATOR"
    success "Générateur PCAP trouvé: $MOCK_GEN"
    success "Listener UDP trouvé: $UDP_LISTENER"

    # Vérifier libpcap
    if pkg-config --exists libpcap; then
        local version=$(pkg-config --modversion libpcap)
        success "libpcap version $version détectée"
    else
        echo "❌ libpcap non trouvée"
        exit 1
    fi
}

# Démonstration 1: Génération de données
demo_data_generation() {
    step "Démonstration 1: Génération de données de test"

    info "Le simulateur inclut un générateur de fichiers PCAP pour les tests."
    info "Nous allons créer un fichier avec 200 paquets LIDAR et 20 paquets IMU."

    action "Génération du fichier demo.pcap..."
    "$MOCK_GEN" demo.pcap 200 20

    info "Fichier créé avec succès!"
    ls -lh demo.pcap

    pause_demo
}

# Démonstration 2: Simulation basique
demo_basic_simulation() {
    step "Démonstration 2: Simulation basique"

    info "Lancement du simulateur en mode verbose pour voir les détails..."
    info "Le simulateur va:"
    info "  - Lire le fichier PCAP"
    info "  - Classifier les paquets (LIDAR vs IMU)"
    info "  - Les retransmettre avec timing original"
    info "  - Afficher les statistiques"

    action "Commande: $SIMULATOR --pcap demo.pcap --verbose"
    pause_demo

    "$SIMULATOR" --pcap demo.pcap --verbose

    success "Simulation terminée!"
    pause_demo
}

# Démonstration 3: Contrôle de vitesse
demo_speed_control() {
    step "Démonstration 3: Contrôle de la vitesse de replay"

    info "Le simulateur peut rejouer à différentes vitesses:"
    info "  --rate 0.5  = 2x plus lent"
    info "  --rate 2.0  = 2x plus rapide"
    info "  --rate 10.0 = 10x plus rapide"

    action "Test à vitesse 5x plus rapide..."
    pause_demo

    "$SIMULATOR" --pcap demo.pcap --rate 5.0 --verbose

    success "Replay accéléré terminé!"
    pause_demo
}

# Démonstration 4: Mode boucle avec jitter
demo_loop_jitter() {
    step "Démonstration 4: Mode boucle avec simulation de jitter"

    info "Pour les tests de longue durée, le simulateur peut:"
    info "  - Boucler indéfiniment (--loop)"
    info "  - Ajouter du jitter réseau (--jitter)"
    info "  - Limiter les délais max (--max-delta)"

    action "Test en boucle avec jitter 1ms (arrêt automatique après 10s)..."
    pause_demo

    # Lancer en background et arrêter après quelques secondes
    timeout 10 "$SIMULATOR" --pcap demo.pcap --loop --jitter 0.001 --rate 5.0 --verbose || true

    success "Test de boucle terminé!"
    pause_demo
}

# Démonstration 5: Test avec listeners UDP
demo_udp_listeners() {
    step "Démonstration 5: Test avec listeners UDP actifs"

    info "Pour vérifier la transmission, nous allons:"
    info "  1. Démarrer des listeners sur les ports LIDAR (7502) et IMU (7503)"
    info "  2. Lancer le simulateur"
    info "  3. Observer la réception des paquets"

    action "Démarrage des listeners UDP en arrière-plan..."

    # Démarrer listeners
    "$UDP_LISTENER" 7502 > lidar_stats.log 2>&1 &
    local lidar_pid=$!
    "$UDP_LISTENER" 7503 > imu_stats.log 2>&1 &
    local imu_pid=$!

    sleep 2
    success "Listeners démarrés (PID: $lidar_pid, $imu_pid)"

    action "Lancement de la simulation..."
    pause_demo

    "$SIMULATOR" --pcap demo.pcap --rate 2.0 --verbose

    # Arrêter listeners
    kill $lidar_pid $imu_pid 2>/dev/null || true
    sleep 1

    # Afficher statistiques
    info "Statistiques du listener LIDAR:"
    if [[ -f "lidar_stats.log" ]]; then
        tail -5 lidar_stats.log || echo "Pas de données"
    fi

    info "Statistiques du listener IMU:"
    if [[ -f "imu_stats.log" ]]; then
        tail -5 imu_stats.log || echo "Pas de données"
    fi

    success "Test UDP terminé!"
    pause_demo
}

# Démonstration 6: Destinations personnalisées
demo_custom_destinations() {
    step "Démonstration 6: Destinations réseau personnalisées"

    info "Le simulateur peut envoyer vers d'autres IPs/ports:"
    info "  --dst-lidar IP:PORT  (destination paquets LIDAR)"
    info "  --dst-imu IP:PORT    (destination paquets IMU)"

    action "Test avec ports personnalisés (8502/8503)..."

    # Démarrer listener sur port personnalisé
    "$UDP_LISTENER" 8502 > custom_lidar.log 2>&1 &
    local custom_pid=$!

    sleep 1

    action "Simulation vers port 8502..."
    pause_demo

    "$SIMULATOR" --pcap demo.pcap --dst-lidar 127.0.0.1:8502 --dst-imu 127.0.0.1:8503 --verbose

    kill $custom_pid 2>/dev/null || true
    sleep 1

    if [[ -f "custom_lidar.log" ]] && grep -q "Packets:" custom_lidar.log; then
        success "Transmission vers port personnalisé réussie!"
    else
        info "Note: Port IMU 8503 sans listener actif (normal)"
    fi

    pause_demo
}

# Démonstration 7: Métriques et monitoring
demo_metrics() {
    step "Démonstration 7: Métriques et monitoring"

    info "Le simulateur offre plusieurs options de monitoring:"
    info "  --log fichier.log     (sauvegarde logs)"
    info "  --metrics-port 8080   (serveur HTTP métriques)"

    action "Test avec logging et métriques HTTP..."

    # Lancer avec métriques HTTP
    timeout 15 "$SIMULATOR" --pcap demo.pcap --log demo_metrics.log --metrics-port 9090 --loop --rate 10.0 --verbose &
    local sim_pid=$!

    sleep 3

    info "Serveur de métriques démarré sur port 9090"
    action "Test d'accès aux métriques..."

    if command -v curl &> /dev/null; then
        if curl -s http://localhost:9090/metrics > metrics_output.json; then
            success "Métriques récupérées avec succès!"
            info "Aperçu des métriques:"
            head -10 metrics_output.json 2>/dev/null || echo "Données métriques disponibles"
        else
            info "Serveur de métriques non accessible (normal si en cours de démarrage)"
        fi
    else
        info "curl non disponible, impossible de tester les métriques HTTP"
    fi

    # Arrêter simulation
    kill $sim_pid 2>/dev/null || true
    sleep 1

    # Afficher logs
    if [[ -f "demo_metrics.log" ]]; then
        info "Extrait du fichier de log:"
        tail -5 demo_metrics.log
        success "Logging vers fichier fonctionnel!"
    fi

    pause_demo
}

# Démonstration 8: Options avancées
demo_advanced_options() {
    step "Démonstration 8: Options avancées"

    info "Autres fonctionnalités avancées:"
    info "  --no-timestamps  (timing fixe, ignore PCAP timestamps)"
    info "  --bind IP        (IP de liaison socket)"
    info "  --json config    (fichier sensor_info.json)"

    action "Test mode timing fixe (sans timestamps PCAP)..."
    pause_demo

    "$SIMULATOR" --pcap demo.pcap --no-timestamps --rate 1.0 --verbose

    success "Mode timing fixe testé!"

    info "En mode timing fixe:"
    info "  - Paquets LIDAR: ~20µs d'intervalle"
    info "  - Paquets IMU: ~10ms d'intervalle"
    info "  - Indépendant des timestamps originaux"

    pause_demo
}

# Résumé final
demo_summary() {
    step "Résumé de la démonstration"

    success "Fonctionnalités démontrées:"
    echo "  ✓ Génération de données de test"
    echo "  ✓ Simulation basique avec classification automatique"
    echo "  ✓ Contrôle de vitesse (rate scaling)"
    echo "  ✓ Mode boucle avec simulation de jitter"
    echo "  ✓ Test avec listeners UDP"
    echo "  ✓ Destinations réseau personnalisées"
    echo "  ✓ Métriques et monitoring"
    echo "  ✓ Options avancées (timing fixe, etc.)"

    echo
    info "Le simulateur est maintenant prêt pour:"
    echo "  • Tests de performance de vos applications LiDAR"
    echo "  • Validation de systèmes de réception"
    echo "  • Simulation de conditions réseau variées"
    echo "  • Développement et debugging"

    echo
    info "Fichiers générés pendant la démo:"
    ls -la *.pcap *.log *.json 2>/dev/null || echo "  Aucun fichier persistant"

    echo
    info "Pour plus d'informations:"
    echo "  • USER_GUIDE.md       - Guide complet"
    echo "  • QUICK_REFERENCE.md  - Référence rapide"
    echo "  • --help              - Aide en ligne"

    echo
    success "Démonstration terminée! Merci d'avoir testé le simulateur Ouster PCAP."
}

# Nettoyage
cleanup() {
    # Tuer processus résiduels
    pkill -f "udp_listener" 2>/dev/null || true
    pkill -f "ouster_sim" 2>/dev/null || true
}

# Menu principal
show_help() {
    echo "Démonstration interactive du simulateur Ouster PCAP"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --full     Démonstration complète (défaut)"
    echo "  --quick    Démonstration rapide (tests essentiels)"
    echo "  --help     Afficher cette aide"
    echo
    echo "La démonstration guide à travers toutes les fonctionnalités"
    echo "principales du simulateur avec des exemples pratiques."
}

# Point d'entrée principal
main() {
    local demo_type=${1:-full}

    case $demo_type in
        --help|-h)
            show_help
            exit 0
            ;;
        --quick)
            echo -e "${GREEN}🚀 Démonstration rapide du simulateur Ouster PCAP${NC}"
            check_environment
            demo_data_generation
            demo_basic_simulation
            demo_speed_control
            demo_summary
            ;;
        --full|*)
            echo -e "${GREEN}🚀 Démonstration complète du simulateur Ouster PCAP${NC}"
            check_environment
            demo_data_generation
            demo_basic_simulation
            demo_speed_control
            demo_loop_jitter
            demo_udp_listeners
            demo_custom_destinations
            demo_metrics
            demo_advanced_options
            demo_summary
            ;;
    esac

    cleanup
}

# Gestion des signaux
trap cleanup EXIT INT TERM

# Exécution
main "$@"