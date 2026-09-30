#!/usr/bin/env bash
# Keeps results private in refs/ops/results, which is public in a public repository.
#
#   bash ops/seal.sh seal ops-out                # the workflow: ops-out/private/ -> encrypted files
#   bash ops/seal.sh open runs/<id> key.pem      # you: -> runs/<id>/private/
#
# The folder is packed, encrypted with a fresh AES-256 key, and that key is encrypted with the RSA
# public key in ops/results-public-key.pem, so only the holder of the private key can open it.
# Without that file the private folder is dropped, never saved in the clear. To make a key pair:
#
#   openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out results-key.pem
#   openssl pkey -in results-key.pem -pubout -out ops/results-public-key.pem
#
# and keep results-key.pem out of the repository.
set -euo pipefail

public_key=ops/results-public-key.pem
oaep=(-pkeyopt rsa_padding_mode:oaep -pkeyopt rsa_oaep_md:sha256)
cipher=(-aes-256-cbc -pbkdf2 -iter 200000)

secret=$(mktemp)
trap 'rm -f "$secret"' EXIT

case "${1:-}" in
  seal)
    dir=$2
    [ -d "$dir/private" ] || exit 0
    if [ ! -f "$public_key" ]; then
      echo "No $public_key, so $dir/private/ is not saved"
      rm -rf "$dir/private"
      exit 0
    fi
    openssl rand -hex 32 >"$secret"
    # Parts under 50 MB, which GitHub accepts without warnings
    tar -czf - -C "$dir" private |
      openssl enc "${cipher[@]}" -salt -pass "file:$secret" |
      split -b 45m -d -a 2 - "$dir/private.tar.gz.enc."
    openssl pkeyutl -encrypt -pubin -inkey "$public_key" "${oaep[@]}" -in "$secret" -out "$dir/private.key.enc"
    rm -rf "$dir/private"
    echo "Encrypted $dir/private/ ($(cat "$dir"/private.tar.gz.enc.* | wc -c) bytes)"
    ;;
  open)
    dir=$2
    key=$3
    openssl pkeyutl -decrypt -inkey "$key" "${oaep[@]}" -in "$dir/private.key.enc" -out "$secret"
    cat "$dir"/private.tar.gz.enc.* | openssl enc -d "${cipher[@]}" -pass "file:$secret" | tar -xzf - -C "$dir"
    echo "Opened $dir/private/"
    ;;
  *)
    echo "usage: bash ops/seal.sh seal <folder> | open <folder> <private key>" >&2
    exit 2
    ;;
esac
