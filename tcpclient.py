# current open ports: 6047-6054
import socket
import sys
import os
import subprocess
import argparse
from tqdm import tqdm
import tempfile
import shutil


def run_rtp_client(server, tcp_port, udp_port, inputs, processed):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (server, int(tcp_port))
    sock.connect(server_address)
    print("connected to " + str(server_address))

    files = os.listdir(inputs)
    dirpath = tempfile.mkdtemp()

    if os.path.exists(processed):
        with open(processed, "r") as proc_f:
            files_done = [line.strip() for line in proc_f]
    else:
        files_done = []

    files = [f for f in files if f not in files_done]

    try:
        for f in tqdm(files):
            wav_id = os.path.join(inputs, f)
            opus_id = os.path.join(dirpath, f.split(".")[0] + ".opus")
            p = subprocess.Popen(
                [os.environ["OPUSENC"], "--expect-loss", "2", wav_id, opus_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            p.wait()

            message = bytes(f.split(".")[0], "utf-8")
            # print(message)
            sock.sendall(message)

            data = sock.recv(80)
            if data == message:
                p = subprocess.Popen(
                    [os.environ["OPUSRTP"], "-d", server, "-p", str(udp_port), opus_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                p.wait()
                sock.sendall(b"ack")
                data = sock.recv(80)
                if data == b"ack":
                    files_done.append(f)
                    continue
                print("Breaking! ack not received. Instead - " + str(data))
                break
            else:
                print(
                    "Breaking due to mismatch: data is - "
                    + str(data)
                    + ", while message is - "
                    + str(message)
                )
                break

    except Exception as e:
        print(e)

    finally:
        shutil.rmtree(dirpath)
        with open(processed, "w") as proc_f:
            proc_f.writelines([line + "\n" for line in files_done])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs")
    parser.add_argument("--server", default="129.97.26.80")
    parser.add_argument("--tcp_port", default=6047)
    parser.add_argument("--udp_port", default=6048)
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.out_dir, "info")):
        with open(os.path.join(args.out_dir, "info"), "w") as f:
            f.write("inputs: " + args.inputs + "\n")
            f.write("server: " + args.server + "\n")

    run_rtp_client(
        args.server,
        args.tcp_port,
        args.udp_port,
        args.inputs,
        os.path.join(args.out_dir, "processed"),
    )
