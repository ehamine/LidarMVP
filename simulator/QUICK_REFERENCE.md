# Référence Rapide - Simulateur Ouster PCAP

## Commandes essentielles

### Démarrage rapide
```bash
# 1. Compiler
./build.sh

# 2. Générer test PCAP
./build/examples/mock_pcap_generator test.pcap

# 3. Tester
./build/ouster_sim --pcap test.pcap --verbose
```

### Génération de données
```bash
# PCAP basique (1000 LIDAR + 100 IMU)
./examples/mock_pcap_generator test.pcap

# PCAP personnalisé
./examples/mock_pcap_generator heavy.pcap 5000 500

# PCAP minimal
./examples/mock_pcap_generator quick.pcap 10 5
```

### Tests avec listeners
```bash
# Terminal 1: Listener LIDAR
./examples/udp_listener 7502

# Terminal 2: Listener IMU
./examples/udp_listener 7503

# Terminal 3: Simulateur
./ouster_sim --pcap test.pcap --verbose
```

## Options principales

### Syntaxe de base
```bash
./ouster_sim --pcap FICHIER [OPTIONS]
```

### Réseau
```bash
--dst-lidar IP:PORT       # Destination LIDAR (défaut: 127.0.0.1:7502)
--dst-imu IP:PORT         # Destination IMU (défaut: 127.0.0.1:7503)
--bind IP                 # IP de liaison (défaut: 0.0.0.0)
--json sensor_info.json   # Config JSON optionnelle
```

### Timing
```bash
--rate 2.0               # 2x plus rapide
--rate 0.5               # 2x plus lent
--loop                   # Boucle infinie
--no-timestamps          # Rate fixe (ignore timestamps PCAP)
--jitter 0.001           # Jitter de 1ms
--max-delta 0.1          # Délai max 100ms
```

### Monitoring
```bash
--verbose                # Logs détaillés
--log fichier.log        # Sauvegarde logs
--metrics-port 8080      # Métriques HTTP
```

## Scénarios courants

### Test de performance
```bash
# Vitesse 5x
./ouster_sim --pcap data.pcap --rate 5.0 --verbose

# Stress test en boucle
./ouster_sim --pcap test.pcap --loop --rate 10.0
```

### Test de robustesse
```bash
# Avec jitter réseau
./ouster_sim --pcap field.pcap --jitter 0.01 --loop

# Rate fixe sans timestamps
./ouster_sim --pcap any.pcap --no-timestamps
```

### Réseau distant
```bash
# Vers autre machine
./ouster_sim --pcap data.pcap \
  --dst-lidar 192.168.1.100:7502 \
  --dst-imu 192.168.1.100:7503
```

### Avec sensor_info
```bash
# Utilise ports du JSON
./ouster_sim --pcap sensor.pcap --json sensor_info.json
```

## Métriques et debugging

### Informations PCAP
```bash
./ouster_sim --pcap file.pcap --verbose | head -10
# Affiche: nombre paquets, durée, estimation LIDAR/IMU
```

### Statistiques finales
```bash
./ouster_sim --pcap test.pcap --verbose | tail -15
# Affiche: paquets envoyés, erreurs, performance
```

### Métriques HTTP
```bash
./ouster_sim --pcap test.pcap --metrics-port 8080 &
curl http://localhost:8080/metrics
```

### Classification debug
```bash
./ouster_sim --pcap unknown.pcap --verbose | grep "Classification:"
# Montre répartition LIDAR/IMU/unknown
```

## Codes de sortie

- **0** : Succès
- **1** : Erreur arguments/fichier
- **2** : Erreur réseau
- **3** : Erreur PCAP

## Signaux

- **Ctrl+C (SIGINT)** : Arrêt propre avec stats finales
- **SIGTERM** : Arrêt immédiat

## Fichiers générés

```bash
test.pcap              # Données PCAP générées
simulation.log         # Logs détaillés (si --log)
```

## Dépannage express

### Erreur "Failed to open PCAP"
```bash
# Vérifier permissions
ls -la fichier.pcap
chmod +r fichier.pcap
```

### Erreur "Permission denied"
```bash
# Utiliser ports > 1024
./ouster_sim --pcap test.pcap --dst-lidar 127.0.0.1:8502
```

### Pas de paquets reçus
```bash
# Vérifier listeners actifs
netstat -ulnp | grep 7502
netstat -ulnp | grep 7503

# Test avec verbose
./ouster_sim --pcap test.pcap --verbose
```

### Performance lente
```bash
# Réduire rate
./ouster_sim --pcap big.pcap --rate 0.1

# Mode sans timestamps
./ouster_sim --pcap big.pcap --no-timestamps
```

## Configuration sensor_info.json

```json
{
  "udp_port_lidar": 7502,
  "udp_port_imu": 7503,
  "udp_dest": "192.168.1.10"
}
```

## Performance typique

- **Débit** : 100K+ pps
- **Latence** : <1µs precision
- **CPU** : <10% utilisation
- **RAM** : <50MB

## Commandes utiles

```bash
# Version et aide
./ouster_sim --version
./ouster_sim --help

# Test complet
./examples/mock_pcap_generator quick.pcap 100 10
./ouster_sim --pcap quick.pcap --verbose --rate 2.0

# Validation installation
pkg-config --exists libpcap && echo "OK" || echo "Installer libpcap-dev"
```