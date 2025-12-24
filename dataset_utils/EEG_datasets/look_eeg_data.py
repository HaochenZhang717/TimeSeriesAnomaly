import os
import numpy as np
import mne


def read_edf(
    edf_path: str,
    channels=None,
    target_sfreq: int | None = None,
    normalize: bool = True,
):
    """
    Read a CHB-MIT EDF file and return EEG data as numpy array.

    Parameters
    ----------
    edf_path : str
        Path to .edf file
    channels : list[str] or None
        Channel names to keep (e.g. ["FP1-F7", "F7-T7"]).
        If None, keep all channels.
    target_sfreq : int or None
        If given, resample to this sampling rate (e.g. 128).
    normalize : bool
        Whether to z-normalize per channel.

    Returns
    -------
    data : np.ndarray
        Shape (T, C), where T = time steps, C = channels
    sfreq : int
        Sampling rate after resampling
    channel_names : list[str]
        Channel names
    """
    if not os.path.exists(edf_path):
        raise FileNotFoundError(edf_path)

    # Read EDF
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Original sampling rate
    sfreq = int(raw.info["sfreq"])

    # Select channels
    if channels is not None:
        raw.pick_channels(channels)

    # Resample if needed
    if target_sfreq is not None and target_sfreq != sfreq:
        raw.resample(target_sfreq)
        sfreq = target_sfreq

    # Get data: shape (C, T)
    data = raw.get_data()

    # Transpose to (T, C)
    data = data.T

    # Per-channel normalization
    if normalize:
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + 1e-8
        data = (data - mean) / std

    channel_names = raw.ch_names

    return data, sfreq, channel_names


if __name__ == "__main__":
    # Example usage
    edf_file = "chb01/chb01_03.edf"

    data, sfreq, ch_names = read_edf(
        edf_file,
        channels=None,       # or ["FP1-F7"]
        target_sfreq=256,    # keep original
        normalize=True,
    )

    print("EDF:", edf_file)
    print("Data shape:", data.shape)   # (T, C)
    print("Sampling rate:", sfreq)
    print("Channels:", ch_names[:5], "...")