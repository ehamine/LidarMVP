1 — Objectif

Fournir une application C++ CLI nommée ouster_sim qui :

    charge un pcap (pcap/pcapng) et le sensor_info.json associé,

    rejoue les paquets UDP contenus dans le pcap en respectant les timestamps d’origine (avec multiplicateur de vitesse --rate),

    sépare les paquets LIDAR et IMU et les envoie vers les ports / IP configurés (par défaut 127.0.0.1:7502 pour lidar, 127.0.0.1:7503 pour IMU),

    peut boucler (--loop), accélérer/ralentir (--rate), forcer les IP/ports cibles (--dst-lidar, --dst-imu), et simuler jitter/noise optionnel,

    expose logs et métriques (paquets/s, paquets envoyés par type, erreurs),

    configurable via CLI et fichier YAML/JSON optionnel.

2 — Contraintes & exigences

    Respecter le contenu binaire des payloads UDP : l’app doit envoyer les bytes tels qu’extraits du pcap (pas de reconstruction).

    Respecter la séparation LIDAR vs IMU : déterminer par le port destination original dans le pcap (ou par heuristique si absent).

    Respecter les timings : utiliser les timestamps absolus du pcap et envoyer avec les deltas ajustés par rate.

    Supporter pcap/pcapng (libpcap).

    Fonctionner sous Linux/Ubuntu (build via CMake).

    Faible latence et minimal jitter : préférer clock_nanosleep si disponible.

    Option Docker pour banc de test.

3 — Dépendances proposées

    C++17 (ou C++20)

    libpcap (libpcap-dev) — lecture pcap/pcapng

    nlohmann/json (header-only) — parse sensor_info.json

    spdlog (optionnel) ou simple logging maison

    CMake

    (optionnel) GoogleTest pour tests unitaires

    (optionnel) boost::program_options ou CLI11 pour parsing CLI

4 — Architecture / modules

    main : parsing CLI + config, initialisation, orchestration.

    PcapReader : wrapper autour de libpcap pour itérer paquets (expose ts, src_ip, dst_ip, src_port, dst_port, payload).

    SensorInfo : parseur JSON pour extraire metadata utiles (num_lasers, beam_angles, sensor_name) — utilisé pour validation et logs.

    PacketClassifier : décide si un paquet est LIDAR ou IMU (par port, fallback heuristique).

    PacketScheduler : gère la logique temporelle (respect du delta/timestamps, rate multiplier, loop) et alimente le Sender.

    SenderUDP : socket UDP performant (non bloquant/buffered), envoi des payloads vers destination.

    MetricsLogger : compte paquets envoyés, erreurs, RTT approximatif, alarmes.

    Config : struct contenant options CLI / fichier.

    Tests : unités pour PcapReader (mock pcap), PacketClassifier, Scheduler timing (simulation).

5 — Format CLI (exemple)

ouster_sim \
  --pcap sample.pcap \
  --json sensor_info.json \
  --dst-lidar 127.0.0.1:7502 \
  --dst-imu 127.0.0.1:7503 \
  --rate 1.0 \
  --loop \
  --jitter 0.01 \
  --bind 0.0.0.0 \
  --verbose

Options clés :

    --pcap (obligatoire)

    --json (obligatoire si disponible)

    --dst-lidar (ip:port)

    --dst-imu (ip:port)

    --rate (float > 0)

    --loop (bool)

    --no-timestamps (envoie à cadence fixe)

    --jitter (s pour simuler jitter gaussien)

    --bind (IP locale pour bind socket)

    --log (path)

6 — Comportement fonctionnel (flow principal)

    Parser args -> charger config.

    Ouvrir et parser sensor_info.json (s’il existe) -> log config capteur.

    Ouvrir pcap via PcapReader.

    Option: pré-scan rapide pour compter paquets et prédire durée (optionnel).

    Créer sockets UDP pour LIDAR/IMU (bind optionnel).

    Boucle lecture :

        pour chaque paquet extrait (avec ts):

            classifier (lidar/imu) -> choisir destination port (override si fourni)

            calculer délai = (ts - last_ts) / rate (si use_timestamps)

            sleep precise (nanosleep) pour attendre délai (si délai > 0)

            envoyer payload via SenderUDP::send(payload, dst)

            metrics++ ; logs si verbose

    Si --loop, remettre last_ts = None et relancer la lecture.

    Arrêt propre sur SIGINT/SIGTERM (fermeture sockets, flush logs).

7 — Détails d’implémentation importants
PcapReader (libpcap)

    Utiliser pcap_open_offline() + pcap_next_ex() pour lire paquet par paquet.

    Obtenir timestamp struct pcap_pkthdr.ts -> convertir en double seconds (sec + usec/1e6) ou std::chrono::nanoseconds.

    Décoder en-têtes Ethernet/IP/UDP pour extraire dst_port, dst_ip :

        skip Ethernet header (14 bytes) si présent, identifier EtherType (0x0800 IPv4, 0x86DD IPv6).

        Pour IPv4: lire IP header ihl for offset to UDP header.

        Extract UDP header (dport, sport), payload pointer.

    Fournir API bool next(Packet &p) avec p.timestamp, p.dst_ip, p.dst_port, p.payload.

PacketClassifier

    Heuristique :

        Si dst_port == 7502 -> LIDAR

        Else if dst_port == 7503 -> IMU

        Else if sensor_info.json contient udp_port_lidar mapping -> use it

        Else fallback: consider large payloads (> 1200 bytes typical for lidar) => lidar; small payloads => imu

    Exposer logs quand classification incertaine.

PacketScheduler & timing

    Stocker last_ts (double seconds)

    delta = (pkt_ts - last_ts) / rate

    If no_timestamps flag => delta = fixed_interval (e.g., 1 / (packets_per_second_est))

    Sleep using clock_nanosleep(CLOCK_MONOTONIC, 0, &timespec, nullptr) for precision.

    Option --max-delta to cap sleeps (si pcap a pause anormale).

    Option --align-wallclock pour démarrer playback aligné à l’heure réelle.

SenderUDP

    Create two UDP sockets (one for lidar, one for imu) or single socket reused.

    Set SO_SNDBUF large, set IP_TOS if needed.

    Use sendto(); optionally non-blocking and check EAGAIN and retry / drop policy.

    Log send errors.

JSON (sensor_info)

    Parse with nlohmann::json.

    Extract beam_azimuth, beam_altitude, udp_port_lidar, udp_port_imu, sensor_name, lidar_mode.

    Validate pcap content against sensor_info (ex: expected packets/scan ratio) — log mismatch warnings.

8 — Robustesse & tests

    Unit tests:

        Mock pcap with synthetic packets -> validate PcapReader extracts payload and ports.

        PacketClassifier with edge cases.

        PacketScheduler timing approximations (simulate timestamps, assert sleep sums).

    Integration test:

        Small pcap with known sequence (2 lidar pkts, 1 imu). Run sim with --dst-lidar localhost:7502, run local UDP listener that records arrival times and compare deltas to expected (within tolerance).

    Stress test:

        Loop pcap à grande vitesse (--rate 5.0), surveiller usage CPU / buffer drops.

    Checkpoint logging: flush metrics every N seconds in JSON.

9 — Logging & métriques

    Log format compact (INFO/WARN/ERROR).

    Metrics every 5s: packets_sent_total, packets_per_sec, avg_send_latency (approx).

    Option --metrics-http-port (optionnel) pour exporter prometheus-lite JSON.

10 — Sécurité & permissions

    Rejouer pcap sur interface physique si tu veux reproduire source IP/port requiert pcap_inject / tcpreplay + privilèges root. Ici on envoie via sockets userspace — OK pour tests locaux.

    Vérifier taille des UDP payloads (<=65507 bytes) ; sinon fragment.

11 — CMake / Build minimal

CMakeLists.txt (esquisse)

cmake_minimum_required(VERSION 3.10)
project(ouster_sim VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Pcap REQUIRED) # custom FindPCAP.cmake or use pkg-config
find_package(Threads REQUIRED)
# include nlohmann_json and spdlog as targets or add_subdirectory
add_executable(ouster_sim src/main.cpp src/pcap_reader.cpp src/sender.cpp src/scheduler.cpp ...)
target_include_directories(ouster_sim PRIVATE ${PCAP_INCLUDE_DIRS})
target_link_libraries(ouster_sim PRIVATE ${PCAP_LIBRARIES} Threads::Threads)

12 — Exemple de squelette de code (simplifié)

    Je te montre un extrait minimal pour PcapReader + loop d’envoi (fichier src/main.cpp) — adapt à ton style.

// main.cpp (extrait)
#include "pcap_reader.h"
#include "sender.h"
#include "json.hpp"
#include <chrono>
#include <thread>

int main(int argc,char**argv){
  Config cfg = parse_cli(argc,argv);
  SensorInfo sinfo = SensorInfo::from_file(cfg.json_path);
  PcapReader reader(cfg.pcap_path);
  SenderUDP sender(cfg.bind_ip);

  double last_ts = -1.0;
  Packet pkt;
  while(reader.next(pkt)){
    bool is_lidar = classify(pkt, sinfo);
    auto dst = is_lidar ? cfg.dst_lidar : cfg.dst_imu;
    if(last_ts >= 0){
      double delta = (pkt.ts - last_ts) / cfg.rate;
      if(delta > 0){
        auto ns = std::chrono::duration<double>(delta);
        std::this_thread::sleep_for(ns);
      }
    }
    sender.send_to(dst.ip, dst.port, pkt.payload.data(), pkt.payload.size());
    last_ts = pkt.ts;
  }
  return 0;
}


14 — Tests pratiques à faire une fois codé

    Récupère un sample.pcap Ouster + sensor_info.json (Ouster public samples).

    Lancer : ./ouster_sim --pcap sample.pcap --json sensor_info.json --dst-lidar 127.0.0.1:7502 --rate 1.0

    Lance un listener nc -u -l 7502 ou un petit script Python qui collecte paquets et imprime timestamp/delta.

    Vérifie que la cadence des paquets reçus concorde avec le pcap (deltas) et que payload lengths correspondent.

    Test --rate 2.0, mesurer que les temps sont divisés par 2.

    Test --loop + monitoring mémoire/CPU sur 1h.

15 — Améliorations futures (roadmap court terme)

    ajouter support direct des rosbag (pour publishers ROS).

    exporter en replay «raw ethernet» (via raw sockets) pour reproduire IP/MAC origine (nécessite privilèges root).

    ajouter plugin pour modifier le udp_profile_lidar (reconfigurer packet layout si besoin).

    UI minimal web pour piloter (start/stop/change-rate).

