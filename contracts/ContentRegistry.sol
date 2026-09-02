// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ContentRegistry
 * @notice Stores tamper-evident SHA-256 fingerprints of discovered web content.
 *
 * This contract does NOT store:
 *   - Face embeddings or biometric data
 *   - Face images
 *   - Private or personally identifiable information
 *
 * It stores only:
 *   - A bytes32 content hash (SHA-256 fingerprint)
 *   - The block timestamp
 *   - An optional non-sensitive source identifier (e.g. domain name)
 */
contract ContentRegistry {
    struct Record {
        bytes32 contentHash;
        uint256 timestamp;
        string sourceId;
    }

    /// @notice Mapping from content hash to its on-chain record.
    mapping(bytes32 => Record) public records;

    /// @notice Emitted when a new record is registered.
    event RecordRegistered(
        bytes32 indexed contentHash,
        uint256 timestamp,
        string sourceId
    );

    /**
     * @notice Register a content fingerprint on-chain.
     * @param contentHash  The SHA-256 fingerprint as bytes32.
     * @param sourceId     A non-sensitive identifier (e.g. domain or platform).
     */
    function registerRecord(bytes32 contentHash, string calldata sourceId) external {
        require(records[contentHash].timestamp == 0, "Record already exists");
        records[contentHash] = Record({
            contentHash: contentHash,
            timestamp: block.timestamp,
            sourceId: sourceId
        });
        emit RecordRegistered(contentHash, block.timestamp, sourceId);
    }

    /**
     * @notice Verify whether a content hash exists on-chain.
     * @param contentHash  The hash to look up.
     * @return exists      True if the hash has been registered.
     * @return timestamp   The block timestamp of registration (0 if not found).
     * @return sourceId    The stored source identifier.
     */
    function verifyRecord(bytes32 contentHash)
        external
        view
        returns (bool exists, uint256 timestamp, string memory sourceId)
    {
        Record memory r = records[contentHash];
        return (r.timestamp != 0, r.timestamp, r.sourceId);
    }
}
