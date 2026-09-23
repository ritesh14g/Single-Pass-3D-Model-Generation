#!/usr/bin/env bash
# Before the GPU box is handed to someone else: find, then remove, the credentials on it.
#
#   bash scripts/box_wipe_credentials.sh          # only reports what it finds
#   bash scripts/box_wipe_credentials.sh --yes    # removes it
#
# Looks for: the Hugging Face token, git credentials (including a token inside a remote URL),
# SSH private keys, cloud/API keys in the shell profile and environment, and Jupyter's saved
# state. It never prints a secret's value — only where it is.
#
# Also offered with --yes: deleting the working copy and its run data, so the flight footage
# and reconstructions do not go to the next user.
set -uo pipefail

APPLY=${1:-}
REPO=$(cd "$(dirname "$0")/.." && pwd)
FOUND=0

hit() { FOUND=$((FOUND + 1)); printf '  [%d] %s\n' "$FOUND" "$*"; }
gone() { printf '      removed\n'; }

echo "== Hugging Face"
for f in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token" "$HOME/.cache/huggingface/stored_tokens"; do
  [ -f "$f" ] && { hit "token file: $f"; [ "$APPLY" = "--yes" ] && rm -f "$f" && gone; }
done
if [ -n "${HF_TOKEN:-}${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  hit "HF token is set in this shell's environment (unset it, and remove it from any profile below)"
fi

echo "== git"
for f in "$HOME/.git-credentials" "$HOME/.config/git/credentials"; do
  [ -f "$f" ] && { hit "saved git credentials: $f"; [ "$APPLY" = "--yes" ] && rm -f "$f" && gone; }
done
if git -C "$REPO" remote -v 2>/dev/null | grep -qE '://[^/@]+@'; then
  hit "the repo's remote URL contains a token or username"
  if [ "$APPLY" = "--yes" ]; then
    for name in $(git -C "$REPO" remote); do
      url=$(git -C "$REPO" remote get-url "$name")
      clean=$(printf '%s' "$url" | sed -E 's#(://)[^/@]+@#\1#')
      git -C "$REPO" remote set-url "$name" "$clean" && printf '      remote %s cleaned\n' "$name"
    done
  fi
fi
helper=$(git -C "$REPO" config --get credential.helper 2>/dev/null || true)
[ -n "$helper" ] && { hit "git credential helper configured: $helper"; \
  [ "$APPLY" = "--yes" ] && git config --global --unset-all credential.helper 2>/dev/null && gone; }

echo "== ssh"
if [ -d "$HOME/.ssh" ]; then
  keys=$(find "$HOME/.ssh" -maxdepth 1 -type f ! -name 'known_hosts*' ! -name '*.pub' ! -name 'config' 2>/dev/null)
  [ -n "$keys" ] && { hit "private keys in ~/.ssh:"; printf '      %s\n' $keys; \
    [ "$APPLY" = "--yes" ] && rm -f $keys && gone; }
fi

echo "== shell profiles and environment"
for f in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile" "$HOME/.zshrc" "$HOME/.netrc" "$HOME/.condarc"; do
  [ -f "$f" ] || continue
  if grep -qiE '(token|secret|api[_-]?key|password|hf_[A-Za-z0-9]|ghp_[A-Za-z0-9]|AKIA)' "$f" 2>/dev/null; then
    hit "possible secret in $f (lines: $(grep -ciE '(token|secret|api[_-]?key|password|hf_|ghp_|AKIA)' "$f"))"
    echo "      edit it by hand; this script will not rewrite your shell profile"
  fi
done
[ -f "$HOME/.netrc" ] && { hit "~/.netrc (stores login details in plain text)"; \
  [ "$APPLY" = "--yes" ] && rm -f "$HOME/.netrc" && gone; }
env | grep -iE '^(HF_|HUGGING|GITHUB_|GH_|AWS_|GOOGLE_|OPENAI_|ANTHROPIC_)[A-Z_]*=' | cut -d= -f1 \
  | while read -r name; do hit "environment variable set: $name"; done

echo "== jupyter and shell history"
for f in "$HOME/.jupyter/jupyter_server_config.json" "$HOME/.local/share/jupyter/runtime" "$HOME/.bash_history" \
         "$HOME/.python_history" "$HOME/.ipython/profile_default/history.sqlite"; do
  [ -e "$f" ] && { hit "$f (may contain tokens you typed)"; [ "$APPLY" = "--yes" ] && rm -rf "$f" && gone; }
done

echo "== project data"
if [ -d "$REPO" ]; then
  size=$(du -sh "$REPO" 2>/dev/null | cut -f1)
  hit "the working copy and its run data: $REPO ($size)"
  if [ "$APPLY" = "--yes" ]; then
    printf '      delete it? the flight footage and reconstructions live here [y/N]: '
    read -r answer < /dev/tty
    case "$answer" in
      [yY]*) rm -rf "$REPO" && gone ;;
      *) echo "      kept" ;;
    esac
  fi
fi

echo
if [ "$APPLY" != "--yes" ]; then
  echo "$FOUND item(s) found. Nothing was changed — re-run with --yes to remove them:"
  echo "    bash scripts/box_collect.sh esri_full2     # download your results FIRST"
  echo "    bash scripts/box_wipe_credentials.sh --yes"
else
  echo "Done. Check the list above for anything that must be edited by hand (shell profiles)."
  echo "Then change any token that was on this machine — treat it as exposed."
fi
