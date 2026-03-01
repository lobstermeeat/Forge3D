/** Common interface for all experience format controllers. */
export interface FormatController {
  update(dt: number): void;
  togglePlayPause(): boolean;
  isPlaying(): boolean;
  dispose(): void;
  getDuration?(): number;
  getCurrentTime?(): number;
  seek?(time: number): void;
}
