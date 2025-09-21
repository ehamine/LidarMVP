# Guide Utilisateur - Simulateur Ouster PCAP

## Vue d'ensemble

Le simulateur Ouster PCAP est un outil haute performance pour rejouer des fichiers PCAP de capteurs LiDAR Ouster avec un contrôle précis du timing et de la configuration réseau.

## Installation et compilation

### Prérequis

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake libpcap-dev pkg-config

# Vérification des dépendances
pkg-config --exists libpcap && echo "libpcap OK" || echo "libpcap manquant"
cmake --version
```

### Compilation

```bash
cd simulator
./build.sh
```

Les exécutables sont créés dans `build/` :
- `./ouster_sim` - Simulateur principal
- `./examples/udp_listener` - Listener UDP pour tests
- `./examples/mock_pcap_generator` - Générateur de PCAP de test

## Guide de démarrage rapide

### 1. Génération d'un fichier de test

```bash
cd build
./examples/mock_pcap_generator test.pcap 1000 100
# Génère test.pcap avec 1000 paquets LIDAR et 100 paquets IMU
```

### 2. Test basique

```bash
./ouster_sim --pcap test.pcap --verbose
```

### 3. Test avec listeners UDP

```bash
# Terminal 1 - Listener LIDAR
./examples/udp_listener 7502

# Terminal 2 - Listener IMU
./examples/udp_listener 7503

# Terminal 3 - Simulateur
./ouster_sim --pcap test.pcap --verbose
```

## Options de configuration

### Options obligatoires

```bash
--pcap PATH        # Fichier PCAP à rejouer
```

### Configuration réseau

```bash
--json PATH                    # Fichier sensor_info.json (optionnel)
--dst-lidar IP:PORT           # Destination paquets LIDAR (défaut: 127.0.0.1:7502)
--dst-imu IP:PORT             # Destination paquets IMU (défaut: 127.0.0.1:7503)
--bind IP                     # IP de liaison socket (défaut: 0.0.0.0)
```

### Contrôle de timing

```bash
--rate FLOAT                  # Multiplicateur de vitesse (défaut: 1.0)
--loop                        # Rejouer en boucle
--no-timestamps              # Ignorer timestamps PCAP, utiliser rate fixe
--jitter FLOAT               # Jitter gaussien en secondes (défaut: 0.0)
--max-delta FLOAT            # Délai maximum entre paquets (défaut: 1.0s)
```

### Logging et monitoring

```bash
--verbose                     # Logs détaillés
--log PATH                    # Fichier de log
--metrics-port PORT           # Port HTTP pour métriques (0=désactivé)
```

## Exemples d'utilisation

### Scénarios de test

#### Test de performance - vitesse élevée
```bash
./ouster_sim --pcap production.pcap --rate 5.0 --verbose
# Rejoue 5x plus vite que l'original
```

#### Test de stabilité - boucle continue
```bash
./ouster_sim --pcap baseline.pcap --loop --jitter 0.001 --log replay.log
# Boucle infinie avec jitter de 1ms
```

#### Test réseau - destinations personnalisées
```bash
./ouster_sim --pcap field_data.pcap \
  --dst-lidar 192.168.1.100:7502 \
  --dst-imu 192.168.1.100:7503 \
  --metrics-port 8080
```

#### Test avec sensor_info.json
```bash
./ouster_sim --pcap sensor_data.pcap --json sensor_info.json --rate 2.0
# Utilise les ports personnalisés du fichier JSON
```

#### Mode timing fixe (sans timestamps)
```bash
./ouster_sim --pcap any_data.pcap --no-timestamps --rate 1.0
# 20µs entre paquets LIDAR, 10ms entre paquets IMU
```

### Génération de données de test

#### PCAP basique
```bash
./examples/mock_pcap_generator basic.pcap
# 1000 LIDAR + 100 IMU (défaut)
```

#### PCAP personnalisé
```bash
./examples/mock_pcap_generator heavy_load.pcap 5000 500
# 5000 paquets LIDAR + 500 paquets IMU
```

#### PCAP minimal
```bash
./examples/mock_pcap_generator minimal.pcap 10 5
# Test rapide avec peu de paquets
```

## Classification des paquets

Le simulateur classifie automatiquement les paquets selon plusieurs méthodes :

### 1. Classification par port (priorité haute)
- Port 7502 → LIDAR
- Port 7503 → IMU

### 2. Classification par sensor_info.json
```json
{
  "udp_port_lidar": 7502,
  "udp_port_imu": 7503
}
```

### 3. Classification heuristique (fallback)
- Paquets > 1000 bytes → LIDAR
- Paquets < 200 bytes → IMU

## Monitoring et métriques

### Logs verbose
```bash
./ouster_sim --pcap test.pcap --verbose
```

Affiche :
- Informations PCAP (nombre de paquets, durée)
- Progression en temps réel
- Statistiques finales détaillées

### Métriques HTTP
```bash
./ouster_sim --pcap test.pcap --metrics-port 8080
```

Accès aux métriques :
```bash
curl http://localhost:8080/metrics
```

### Sauvegarde des logs
```bash
./ouster_sim --pcap test.pcap --verbose --log simulation.log
```

## Gestion des erreurs courantes

### "Failed to open PCAP file"
- Vérifier les permissions du fichier
- Vérifier que libpcap est installé
- Utiliser un chemin absolu

### "Permission denied" sur binding
- Utiliser des ports > 1024 pour éviter les permissions root
- Vérifier que les ports ne sont pas déjà utilisés

### Perte de paquets élevée
- Réduire le rate : `--rate 0.5`
- Augmenter les buffers système
- Vérifier la charge CPU/réseau

### Timing imprécis
- Utiliser `--no-timestamps` pour rate fixe
- Réduire le jitter : `--jitter 0`
- Éviter la charge système élevée

## Performance

### Capacité typique
- **Débit** : >100,000 paquets/seconde
- **Précision timing** : Sub-microseconde (avec clock_nanosleep)
- **Mémoire** : <50MB pour gros fichiers PCAP
- **CPU** : <10% en utilisation normale

### Optimisation
- Utiliser `--no-timestamps` pour performance maximale
- Réduire le logging en production
- Ajuster les buffers UDP si nécessaire

## Intégration dans les workflows

### Script de test automatisé
```bash
#!/bin/bash
# Génération de données
./examples/mock_pcap_generator test_data.pcap 2000 200

# Démarrage listeners
./examples/udp_listener 7502 > lidar_stats.log &
LIDAR_PID=$!
./examples/udp_listener 7503 > imu_stats.log &
IMU_PID=$!

# Simulation
./ouster_sim --pcap test_data.pcap --rate 2.0 --log sim.log

# Nettoyage
kill $LIDAR_PID $IMU_PID
```

### Test de stress réseau
```bash
# Test haute fréquence
./ouster_sim --pcap heavy.pcap --rate 10.0 --loop --jitter 0.0001

# Test de robustesse
./ouster_sim --pcap field.pcap --loop --jitter 0.01 --max-delta 0.1
```

### Validation de systèmes
```bash
# Test avec vrais ports Ouster
./ouster_sim --pcap real_sensor.pcap \
  --dst-lidar 192.168.1.10:7502 \
  --dst-imu 192.168.1.10:7503 \
  --rate 1.0 --verbose
```

## Support et debugging

### Informations système
```bash
./ouster_sim --version
pkg-config --modversion libpcap
```

### Debug détaillé
```bash
./ouster_sim --pcap debug.pcap --verbose --log debug.log --jitter 0
# Analyse du fichier debug.log pour diagnostic
```

### Validation des données
```bash
# Vérifier classification
./ouster_sim --pcap unknown.pcap --verbose | grep "Classification:"

# Tester avec données minimales
./examples/mock_pcap_generator debug.pcap 10 2
./ouster_sim --pcap debug.pcap --verbose
```

## Limitations connues

- **IPv4 uniquement** : Support IPv6 prévu
- **UDP uniquement** : Pas de support TCP
- **Dépendance Linux** : Précision optimale avec clock_nanosleep
- **Validation capteur** : Nécessite données réelles pour validation complète

## Prochaines étapes

Pour des besoins avancés, consulter :
- `README.md` - Architecture technique
- `project_status.md` - Roadmap développement
- Code source dans `src/` pour personnalisations