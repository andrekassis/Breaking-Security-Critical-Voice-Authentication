#!/usr/bin/python3

# currentopen ports: 6047-6054
import socket
import sys
import subprocess
import os, signal
import time
import argparse
import tempfile
import shutil


def recv_utt(
    data, connection, pcap_dirpath, opus_dirpath, wav_dirpath, port, interface
):
    utt_id = str(data.decode())
    pcap_path = os.path.join(pcap_dirpath, utt_id + ".pcap")
    opus_path = os.path.join(opus_dirpath, utt_id + ".opus")
    wav_path = os.path.join(wav_dirpath, utt_id + ".wav")
    recv_cmd = [
        "tcpdump",
        "-i",
        interface,
        "udp",
        "port",
        str(port),
        "--immediate-mode",
        "-w",
        pcap_path,
    ]
    rtp_cmd = [
        os.environ["OPUSRTP"],
        "--quiet",
        "-c",
        "1",
        "-r",
        "16000",
        "-e",
        pcap_path,
        "-o",
        opus_path,
    ]
    dec_cmd = [
        os.environ["OPUSDEC"],
        "--quiet",
        "--packet-loss",
        "2",
        opus_path,
        wav_path,
    ]

    p = subprocess.Popen(
        recv_cmd,
        shell=False,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    connection.sendall(data)
    data = connection.recv(80)

    if data == b"ack":
        pid = subprocess.Popen(["kill", "-2", str(p.pid)])
        pid.wait()

        while not os.path.exists(pcap_path):
            time.sleep(0.05)

        while not os.path.exists(opus_path):
            pid = subprocess.Popen(
                rtp_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            pid.wait()

        pid = subprocess.Popen(dec_cmd)
        pid.wait()

        connection.sendall(b"ack")
    return data


def run_rtp_server(tcp_port, udp_port, interface, wav_dirpath):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ("0.0.0.0", int(tcp_port))
    print("starting up on " + str(server_address))
    sock.bind(server_address)
    sock.listen(1)

    try:
        connection, client_address = sock.accept()
        pcap_dirpath = tempfile.mkdtemp()
        opus_dirpath = tempfile.mkdtemp()
        print("connection from" + str(client_address))

        while True:
            data = connection.recv(80)
            # print("received " + str(data.decode()))

            if data:
                data = recv_utt(
                    data,
                    connection,
                    pcap_dirpath,
                    opus_dirpath,
                    wav_dirpath,
                    udp_port,
                    interface,
                )
                if not data:
                    break
            else:
                break
    except Exception as e:
        print(e)

    finally:
        connection.close()
        shutil.rmtree(pcap_dirpath)
        shutil.rmtree(opus_dirpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir")
    parser.add_argument("--tcp_port", default=6047)
    parser.add_argument("--udp_port", default=6048)
    parser.add_argument("--interface", default="eno1")
    args = parser.parse_args()
    os.makedirs(args.out_dir)
    run_rtp_server(args.tcp_port, args.udp_port, args.interface, args.out_dir)
